/*
All the classes definitions.
*/

/** 
 * convenience for converting JSON color to rgba that canvas wants
 * Be nice to handle different forms (e.g. no alpha, CSS style, etc.)
 */ 
function getCanvasColor(color){
	return "rgba(" + color.r + "," + color.g + "," + color.b + "," + color.a + ")"; 
}


var _ScatterDataSubset = Backbone.Model.extend({
	defaults: {
		data: null, // an array of objects
	},
	// base model for data points 
	initialize: function(options){
		if (options === undefined) {options = {}}
		_.defaults(options, this.defaults)
		_.defaults(this, options)

		this.n = this.data.length;

		// generate arrays of positions
		this.indices = new Uint32Array( this.n );
		this.positions = new Float32Array( this.n * 3 );

		for (var i = this.data.length - 1; i >= 0; i--) {
			this.indices[i] = i; 
			this.positions[ i*3 ] = this.data[i].x;
			this.positions[ i*3+1 ] = this.data[i].y;
			this.positions[ i*3+2 ] = this.data[i].z;
		};
	},

	getAttr: function(metaKey){
		// return an array of attributes 
		return _.map(this.data, function(record){ return record[metaKey]; });
	},

});


var Scatter3dCloud = Backbone.View.extend({
	// this is the view for points of a single shape
	model: _ScatterDataSubset,

	defaults: {
		texturePath: null,
		data: null, // expect data to be an array of objects
		labelKey: 'sig_id',
	},

	initialize: function(options){
		if (options === undefined) {options = {}}
		_.defaults(options, this.defaults)
		_.defaults(this, options)
		//
		this.setUpScatterGeometry();
	},

	loadMaterial: function(){
		// not in use...
	    var self = this;
	    if (this.texturePath){

	    	var loader = new THREE.TextureLoader();
	    	
	    	loader.load(this.texturePath, function(texture){

				var material = new THREE.PointsMaterial({ 
					vertexColors: THREE.VertexColors,
					size: 2, 
					// sizeAttenuation: false, 
					map: texture, 
					alphaTest: 0.5, 
					transparent: true,
					opacity: 0.6
					});
				self.material = material;
				self.trigger('textureLoaded');
				// console.log('textureLoaded')
	    	});
	    } else{
			var material = new THREE.PointsMaterial({
				vertexColors: THREE.VertexColors,
				size: 2,
				// sizeAttenuation: false, 
				opacity: 0.6,
				transparent: true,
			});
			self.material = material;
			self.trigger('textureLoaded');
	    }		
	},

	setUpScatterGeometry: function(){
		var model = this.model;

		this.geometry = new THREE.BufferGeometry();
		this.geometry.setIndex( new THREE.BufferAttribute( model.indices, 1 ) );
		this.geometry.addAttribute( 'position', new THREE.BufferAttribute( model.positions, 3 ) );		
		this.geometry.addAttribute( 'label', new THREE.BufferAttribute( model.getAttr(this.labelKey), 1 ) );

	    this.geometry.computeBoundingSphere();

	    if (this.texturePath){
	    	var texture = new THREE.TextureLoader().load(this.texturePath)
			var material = new THREE.PointsMaterial({ 
				vertexColors: THREE.VertexColors,
				size: 2, 
				// sizeAttenuation: false, 
				map: texture, 
				alphaTest: 0.5, 
				transparent: true,
				opacity: 0.6
				});
	    } else{
			var material = new THREE.PointsMaterial({
				vertexColors: THREE.VertexColors,
				size: 2,
				// sizeAttenuation: false, 
				opacity: 0.6,
				transparent: true,
			});
	    }
		this.points = new THREE.Points( this.geometry, material );
	},

	setColors: function(colorScale, metaKey){
		// Color points by a certain metaKey given colorScale
		var metas = this.model.getAttr(metaKey)
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
	},

});


var ScatterData = Backbone.Model.extend({
	// model for the data (positions) and metadata. 
	defaults: {
		url: 'toy',
		n: 100, // Number of data points to retrieve
		metaKeys: [],
		data: [], // store data
	},

	url: function(){
		return this.attributes.url + '?n=' + this.n;
	},

	parse: function(response){
		// called whenever a model's data is returned by the server
		nPoints = response.length;
		var metaKeys = [];
		xyz = ['x', 'y', 'z'];
		for (var key in response[0]){
			if (xyz.indexOf(key) === -1){ 
				metaKeys.push(key);
			}
		}
		this.metaKeys = metaKeys;
		this.data = response;
	},

	initialize: function(options){
		// called on construction
		if (options === undefined) {options = {}}
		_.defaults(options, this.defaults)
		_.defaults(this, options)
		// fetch json data from server
		this.fetch()
	},

	groupBy: function(metaKey){
		// group by a metaKey and return an object of _ScatterDataSubset objects keyed by metaKey
		var dataSubsets = _.groupBy(this.data, metaKey);
		var scatterDataSubsets = _.mapObject(dataSubsets, function(records, key){
			return new _ScatterDataSubset({data: records});
		});
		return scatterDataSubsets;
	},

	getAttr: function(metaKey){
		// return an array of attributes 
		return _.map(this.data, function(record){ return record[metaKey]; });
	},

});


var Scatter3dView = Backbone.View.extend({
	// this is the view for all points
	model: ScatterData,

	defaults: {
		WIDTH: window.innerWidth,
		HEIGHT: window.innerHeight,
		DPR: window.devicePixelRatio,
		container: document.body,
		labelKey: 'sig_id', // which metaKey to use as labels
		colorKey: 'dose', // which metaKey to use as colors
		shapeKey: 'cell',
		texturePath: null,
		clouds: [], // to store Scatter3dCloud objects
		textureBasePath: '../lib/textures/d3-symbols/',
	},

	initialize: function(options){
		if (options === undefined) {options = {}}
		_.defaults(options, this.defaults)
		_.defaults(this, options)

		this.listenTo(this.model, 'sync', function(){
			this.setUpStage();

			this.shapeBy(this.shapeKey);


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

	clearScene: function(){
		// remove everythin in the scene
		var scene = this.scene;
		_.each(scene.children, function( object ) {
			scene.remove(object);
		});
	},

	shapeBy: function(metaKey){
		// groupBy the model and init clouds
		// update shapeKey
		this.shapeKey = metaKey;
		// re-coloring nodes
		this.colorBy(this.colorKey);
		// clear this.clouds
		this.clouds = [];
		this.clearScene();

		var scatterDataSubsets = this.model.groupBy(metaKey);
		var textureBasePath = this.textureBasePath;
		var texturePaths = _.map(d3.svg.symbolTypes, function(name){ return textureBasePath + name + '.png';});
		// var texturePaths = [null,
		// '../lib/textures/sprites/circle.png', 
		// '../lib/textures/sprite1.png',
		// '../lib/textures/sprite2.png'];
		var i = 0;
		var nTextures = 0; // counter for number of textures needed to be loaded
		
		for (var key in scatterDataSubsets){
			var texturePath = texturePaths[i % texturePaths.length];
			if (texturePath) { nTextures ++ }
			var cloud = new Scatter3dCloud({
				model: scatterDataSubsets[key],
				texturePath: texturePath, 
			});

			this.clouds.push(cloud)
			this.scene.add( cloud.points );	
			
			i ++;
		}

	},

	renderScatter: function(){
		// update the picking ray with the camera and mouse position
		this.raycaster.setFromCamera( this.mouse, this.camera );

		// calculate objects intersecting the picking ray
		// var intersects = this.raycaster.intersectObject( this.points );
		var allPoints = _.map(this.clouds, function(obj){ return obj.points; });
		var intersects = this.raycaster.intersectObjects( allPoints );

		// reset colors
		this.colorBy(this.colorKey);

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
			var geometry = intersect.object.geometry;
			
			// change color of the point
			geometry.attributes.color.needsUpdate = true;

			geometry.attributes.color.array[idx*3] = 0.1;
			geometry.attributes.color.array[idx*3+1] = 0.8;
			geometry.attributes.color.array[idx*3+2] = 0.1;
			// geometry.computeBoundingSphere();
			// intersect.object.updateMatrix();

			// find the position of the point
			var pointPosition = { 
			    x: geometry.attributes.position.array[idx*3],
			    y: geometry.attributes.position.array[idx*3+1],
			    z: geometry.attributes.position.array[idx*3+2],
			}

			// add text canvas
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
			x: ((pv.x + 1) / 2 * this.WIDTH), // * this.DPR, 
			y: -((pv.y - 1) / 2 * this.HEIGHT), // * this.DPR
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
		// update colorKey
		this.colorKey = metaKey;

		var metas = this.model.getAttr(metaKey);
		var uniqueCats = new Set(metas);
		var nUniqueCats = uniqueCats.size;
		// console.log(uniqueCats, nUniqueCats)

		if (nUniqueCats < 11){
			var colorScale = d3.scale.category10().domain(uniqueCats);
		} else {
			var colorScale = d3.scale.category20().domain(uniqueCats);
		}

		for (var i = this.clouds.length - 1; i >= 0; i--) {
			var cloud = this.clouds[i];
			cloud.setColors(colorScale, metaKey)
		};		
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

var Legends = Backbone.View.extend({
	// A view for the legends of the Scatter3dView
	tagName: 'div',
	defaults: {
		scatterPlot: Scatter3dView,
		w: 400,
		h: 400,
	},

	initialize: function(options){
		if (options === undefined) {options = {}}
		_.defaults(options, this.defaults)
		_.defaults(this, options)

		// this.listenTo
		this.el = document.createElement(this.tagName);

	},

	render: function(){

	},

});




