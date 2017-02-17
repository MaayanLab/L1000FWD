/*
All the classes definitions.
*/

// 
/** 
 * convenience for converting JSON color to rgba that canvas wants
 * Be nice to handle different forms (e.g. no alpha, CSS style, etc.)
 */ 
function getCanvasColor(color){
	return "rgba(" + color.r + "," + color.g + "," + color.b + "," + color.a + ")"; 
}

var ScatterData = Backbone.Model.extend({
	// model for the data (positions) and metadata. 
	defaults: {
		url: 'toy',
		n: 100, // Number of data points to retrieve
		meta: {}, // store metadata
	},

	url: function(){
		return this.attributes.url + '?n=' + this.n;
	},

	parse: function(response){
		// called whenever a model's data is returned by the server
		// fill metadata
		console.log(response)
		nPoints = response.x.length;
		metaKeys = [];
		xyz = ['x', 'y', 'z'];
		for (var key in response){
			if (xyz.indexOf(key) === -1){ 
				metaKeys.push(key);
				this.meta[key] = response[key];
			}
		}

		// fill arrays with data in response
		for (var i = response.x.length - 1; i >= 0; i--) {
			this.indices[i] = i;
			this.positions[ i*3 ] = response.x[i];
			this.positions[ i*3 +1 ] = response.y[i];
			this.positions[ i*3 +2 ] = response.z[i];
		};
	},

	initialize: function(options){
		// called on construction
		if (options === undefined) {options = {}}
		_.defaults(options, this.defaults)
		_.defaults(this, options)
		// init arrays for points
		this.positions = new Float32Array( this.n * 3 );
		this.indices = new Uint32Array( this.n );

		// fetch json data from server
		this.fetch()

	},

});


var Scatter3dView = Backbone.View.extend({
	model: ScatterData,

	defaults: {
		WIDTH: window.innerWidth,
		HEIGHT: window.innerHeight,
		DPR: window.devicePixelRatio,
		container: document.body,
		labelKey: 'sig_id', // which metaKey to use as labels

	},

	initialize: function(options){
		if (options === undefined) {options = {}}
		_.defaults(options, this.defaults)
		_.defaults(this, options)

		this.listenTo(this.model, 'sync', function(){
			this.setUpStage();
			this.setUpScatterGeometry();
			this.colorBy('dose');
			// this.sizeBy('dose');

			this.renderScatter();

		});

	},

	setUpStage: function(){
		// set up THREE.js visualization components
		this.aspectRatio = this.WIDTH / this.HEIGHT;
		
		// set up scene, camera, renderer
		this.scene = new THREE.Scene();
		this.scene.fog = new THREE.FogExp2( 0xcccccc, 0.002 );

		this.renderer = new THREE.WebGLRenderer();
		this.renderer.setClearColor( this.scene.fog.color );
		this.renderer.setPixelRatio( this.DPR );
		this.renderer.setSize( this.WIDTH, this.HEIGHT );

		this.camera = new THREE.PerspectiveCamera( 45, this.aspectRatio, 1, 1000 );
		this.camera.position.z = 20;

		// Put the renderer's DOM into the container
		this.renderer.domElement.id = "renderer";
		this.container.appendChild( this.renderer.domElement );

		// set up orbit controls
		controls = new THREE.OrbitControls( this.camera, this.renderer.domElement );
		var self = this;
		controls.addEventListener( 'change', function(){
			self.renderScatter()
		} );
		controls.enableZoom = true;

		// set up raycaster, mouse
		this.raycaster = new THREE.Raycaster();
		this.raycaster.params.Points.threshold = 0.5;
		this.mouse = new THREE.Vector2();

		// mousemove event
		$(document).on( 'mousemove', function(event){
			// update mouse position
			self.mouse.x = ( event.clientX / self.WIDTH ) * 2 - 1;
			self.mouse.y = - ( event.clientY / self.HEIGHT ) * 2 + 1;

			self.renderScatter();

		});
		
	},


	setUpScatterGeometry: function(){
		var model = this.model;

		this.geometry = new THREE.BufferGeometry();
		this.geometry.setIndex( new THREE.BufferAttribute( model.indices, 1 ) );
		this.geometry.addAttribute( 'position', new THREE.BufferAttribute( model.positions, 3 ) );
		
		this.geometry.addAttribute( 'label', new THREE.BufferAttribute( model.meta[this.labelKey], 1 ) );


		var normals = new Int16Array( model.n * 9 );




		this.geometry.addAttribute( 'normal', new THREE.BufferAttribute( normals, 3 ) );

	    this.geometry.computeBoundingSphere();


		var material = new THREE.PointsMaterial({
			vertexColors: THREE.VertexColors,
			size: 2,
			opacity: 0.6,
			transparent: true,
		});

		this.points = new THREE.Points( this.geometry, material );
		this.scene.add( this.points );

	},

	renderScatter: function(){
		// update the picking ray with the camera and mouse position
		this.raycaster.setFromCamera( this.mouse, this.camera );

		// calculate objects intersecting the picking ray
		var intersects = this.raycaster.intersectObject( this.points );

		var geometry = this.points.geometry;

		geometry.attributes.color.needsUpdate = true;

		// reset colors
		geometry.attributes.color.array = this.colors.slice();
		geometry.computeBoundingSphere();
		this.points.updateMatrix();

		// remove text-label if exists
		var textLabel = document.getElementById('text-label')
		if (textLabel){
		    textLabel.remove();
		}

		// add interactivities if there is intesecting points
		if ( intersects.length > 0 ) {
			// console.log(intersects)
			// only highlight the closest object
			var intersect = intersects[0];
			// console.log(intersect)
			var idx = intersect.index;
			// change color of the point
			geometry.attributes.color.array[idx*3] = 0.1;
			geometry.attributes.color.array[idx*3+1] = 0.8;
			geometry.attributes.color.array[idx*3+2] = 0.1;
			// add text canvas

			// find the position of the point
			var pointPosition = { 
			    x: geometry.attributes.position.array[idx*3],
			    y: geometry.attributes.position.array[idx*3+1],
			    z: geometry.attributes.position.array[idx*3+2],
			}


			var textCanvas = this.makeTextCanvas( geometry.attributes.label.array[idx], 
			    pointPosition.x, pointPosition.y, pointPosition.z,
			    { fontsize: 24, fontface: "Ariel", textColor: {r:0, g:0, b:255, a:1.0} }); 


			textCanvas.id = "text-label"
			this.container.appendChild(textCanvas);

			// geometry.computeBoundingSphere();
		}

		this.renderer.render( this.scene, this.camera );
	},

	makeTextCanvas: function(message, x, y, z, parameters){

		if ( parameters === undefined ) parameters = {}; 
		var fontface = parameters.hasOwnProperty("fontface") ?  
			parameters["fontface"] : "Arial";      
		var fontsize = parameters.hasOwnProperty("fontsize") ?  
			parameters["fontsize"] : 18; 
		var textColor = parameters.hasOwnProperty("textColor") ? 
			parameters["textColor"] : { r:0, g:0, b:255, a:1.0 }; 

		var canvas = document.createElement('canvas'); 
		var context = canvas.getContext('2d'); 

		canvas.width = this.WIDTH; 
		canvas.height = this.HEIGHT; 

		context.font = fontsize + "px " + fontface; 
		context.textBaseline = "alphabetic"; 

		context.textAlign = "left"; 
		// get size data (height depends only on font size) 
		var metrics = context.measureText( message ); 
		var textWidth = metrics.width; 

		// text color.  Note that we have to do this AFTER the round-rect as it also uses the "fillstyle" of the canvas 
		context.fillStyle = getCanvasColor(textColor); 

		// calculate the project of 3d point into 2d plain
		var point = new THREE.Vector3(x, y, z);
		var pv = new THREE.Vector3().copy(point).project(this.camera);
		var coords = {
			x: ((pv.x + 1) / 2 * this.WIDTH) * this.DPR, 
			y: -((pv.y - 1) / 2 * this.HEIGHT) * this.DPR
		};
		// draw the text
		context.fillText(message, coords.x, coords.y)

		// styles of canvas element
		canvas.style.left = 0;
		canvas.style.top = 0;
		canvas.style.position = 'absolute';
		canvas.style.pointerEvents = 'none';

		return canvas;
	},

	colorBy: function(metaKey){
		// Color points by a certain metaKey
		var metas = this.model.meta[metaKey]
		var uniqueCats = new Set(metas)
		var nUniqueCats = uniqueCats.size;
		console.log(uniqueCats, nUniqueCats)

		if (nUniqueCats < 11){
			var colorScale = d3.scale.category10().domain(uniqueCats);
		} else {
			var colorScale = d3.scale.category20().domain(uniqueCats);
		}
		// construct colors BufferAttribute
		var colors = new Float32Array( this.model.n * 3);
		for (var i = metas.length - 1; i >= 0; i--) {
			var color = colorScale(metas[i]);
			color = new THREE.Color(color);
			color.toArray(colors, i*3)
		};

		this.colors = colors;

		this.geometry.addAttribute( 'color', new THREE.BufferAttribute( colors.slice(), 3 ) );
		this.geometry.attributes.color.needsUpdate = true;
		
		this.renderer.render( this.scene, this.camera );
	},

	// sizeBy: function(metaKey){
	// 	// Size points by a certain metaKey
	// 	var metas = this.model.meta[metaKey]
	// 	var sizeExtent = d3.extent(metas)
	// 	var sizeScale = d3.scale.linear()
	// 		.domain(sizeExtent)
	// 		.range([0.1, 4]);
	// 	// construct sizes BufferAttribute
	// 	var sizes = _.map(sizeScale, metas); 

	// 	this.sizes = sizes;

	// 	this.geometry.addAttribute( 'size', new THREE.BufferAttribute( sizes, 1 ) );
	// 	this.geometry.attributes.size.needsUpdate = true;

	// 	this.renderer.render( this.scene, this.camera );
	// }

});


