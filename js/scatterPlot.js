/*
The models and views for the scatter plot.
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
		texture: null, // the THREE.Texture instance
		data: null, // expect data to be an array of objects
		labelKey: 'sig_id',
		pointSize: 0.01,
		sizeAttenuation: true, // true for 3d, false for 2d
	},

	initialize: function(options){
		if (options === undefined) {options = {}}
		_.defaults(options, this.defaults)
		_.defaults(this, options)
		//
		this.setUpScatterGeometry();
	},

	setUpScatterGeometry: function(){
		var model = this.model;

		this.geometry = new THREE.BufferGeometry();
		this.geometry.setIndex( new THREE.BufferAttribute( model.indices, 1 ) );
		this.geometry.addAttribute( 'position', new THREE.BufferAttribute( model.positions, 3 ) );		
		this.geometry.addAttribute( 'label', new THREE.BufferAttribute( model.getAttr(this.labelKey), 1 ) );

	    this.geometry.computeBoundingSphere();

	    var texture = this.texture;
	    if (texture){
			var material = new THREE.PointsMaterial({ 
				vertexColors: THREE.VertexColors,
				size: this.pointSize, 
				sizeAttenuation: this.sizeAttenuation, 
				map: texture, 
				alphaTest: 0.5, 
				transparent: true,
				opacity: 0.6
				});
	    } else{
			var material = new THREE.PointsMaterial({
				vertexColors: THREE.VertexColors,
				size: 0.1,
				sizeAttenuation: this.sizeAttenuation, 
				alphaTest: 0.5, 
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

		// this.colors = colors;

		this.geometry.addAttribute( 'color', new THREE.BufferAttribute( colors.slice(), 3 ) );
		this.geometry.attributes.color.needsUpdate = true;
	},

});


var ScatterData = Backbone.Model.extend({
	// model for the data (positions) and metadata. 
	defaults: {
		url: 'toy',
		n: 100, // Number of data points to retrieve, or number of data points retrieved
		metas: [], // store information about meta [{name: 'metaKey', nUnique: nUnique, type: type}]
		data: [], // store data
	},

	url: function(){
		return this.attributes.url + '?n=' + this.n;
	},

	parse: function(response){
		// called whenever a model's data is returned by the server
		this.n = response.length;
		var xyz = ['x', 'y', 'z'];
		for (var key in response[0]){
			if (xyz.indexOf(key) === -1){ 
				var nUnique = _.unique(_.pluck(response, key)).length;
				var type = typeof response[0][key];
				this.metas.push({
					name: key,
					nUnique: nUnique,
					type: type,
				});
			}
		}
		this.data = response;
	},

	initialize: function(options){
		// called on construction
		if (options === undefined) {options = {}}
		_.defaults(options, this.defaults)
		_.defaults(this, options)
		// fetch json data from server
		this.fetch();
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
		clouds: [], // to store Scatter3dCloud objects
		textures: null, // the Textures collection instance
		pointSize: 0.01, // the size of the points
		showStats: false, // whether to show Stats
		is3d: true, // 3d or 2d
	},

	initialize: function(options){
		if (options === undefined) {options = {}}
		_.defaults(options, this.defaults)
		_.defaults(this, options)

		var self = this;
		this.listenToOnce(this.textures, 'allLoaded', function(){

			self.listenTo(self.model, 'sync', function(){
				console.log('model synced')
				self.setUpStage();
				// self.colorBy(self.colorKey);
				self.shapeBy(self.shapeKey);

			});
		});
	},

	setUpStage: function(){
		// set up THREE.js visualization components
		this.aspectRatio = this.WIDTH / this.HEIGHT;
		
		// set up scene, camera, renderer
		this.scene = new THREE.Scene();
		// this.scene.fog = new THREE.FogExp2( 0xcccccc, 0.002 );

		this.renderer = new THREE.WebGLRenderer();
		// this.renderer.setClearColor( this.scene.fog.color );
		this.renderer.setClearColor( 0xcccccc );
		this.renderer.setPixelRatio( this.DPR );
		this.renderer.setSize( this.WIDTH, this.HEIGHT );

		if (this.is3d){
			this.camera = new THREE.PerspectiveCamera( 70, this.aspectRatio, 0.01, 1000000 );
			this.camera.position.z = this.pointSize * 120;
		} else { // 2d
			ORTHO_CAMERA_FRUSTUM_HALF_EXTENT = 10;
			var left = -ORTHO_CAMERA_FRUSTUM_HALF_EXTENT;
			var right = ORTHO_CAMERA_FRUSTUM_HALF_EXTENT;
			var bottom = -ORTHO_CAMERA_FRUSTUM_HALF_EXTENT;
			var top = ORTHO_CAMERA_FRUSTUM_HALF_EXTENT;
			// Scale up the larger of (w, h) to match the aspect ratio.
			var aspectRatio = this.aspectRatio;
			if (aspectRatio > 1) {
				left *= aspectRatio;
				right *= aspectRatio;
			} else {
				top /= aspectRatio;
				bottom /= aspectRatio;
			}
			this.camera = new THREE.OrthographicCamera( left, right, top, bottom, -1000, 1000 );
		}

		// Put the renderer's DOM into the container
		this.renderer.domElement.id = "renderer";
		this.container.appendChild( this.renderer.domElement );

		var self = this;
		// set up orbit controls
		this.controls = new THREE.OrbitControls( this.camera, this.renderer.domElement );
		this.controls.addEventListener( 'change', function(){
			self.renderScatter()
		} );
		this.controls.enableZoom = true;
		
		if (!this.is3d){
			this.controls.mouseButtons.ORBIT = null;
			this.controls.enableRotate = false;
			this.controls.mouseButtons.PAN = THREE.MOUSE.LEFT;			
		}
		// this.controls.dampingFactor = 0.5;

		// this.controls = new THREE.TrackballControls( this.camera );
		// this.controls.rotateSpeed = 1.0;
		// this.controls.zoomSpeed = 1.2;
		// this.controls.panSpeed = 0.8;
		// this.controls.noZoom = false;
		// this.controls.noPan = false;
		// this.controls.staticMoving = true;
		// this.controls.dynamicDampingFactor = 0.3;

		// set up raycaster, mouse
		this.raycaster = new THREE.Raycaster();
		if (this.is3d){
			this.raycaster.params.Points.threshold = this.pointSize/5;	
		} else {
			this.raycaster.params.Points.threshold = this.pointSize/1000;	
		}
		// this.raycaster.params.Points.threshold = 0.5;
		this.mouse = new THREE.Vector2();

		if (this.showStats) {
			this.stats = new Stats();
			this.container.appendChild( this.stats.dom );
		}

		// mousemove event
		$(document).on( 'mousemove', function(event){
			// update mouse position
			self.mouse.x = ( event.clientX / self.WIDTH ) * 2 - 1;
			self.mouse.y = - ( event.clientY / self.HEIGHT ) * 2 + 1;

			self.renderScatter();

		});
		// window resize event
		$(window).on( 'resize', function(event){
			self.WIDTH = $(self.container).width(); 
			self.HEIGHT = $(self.container).height(); 
			self.camera.aspect = self.WIDTH / self.HEIGHT;
			self.camera.updateProjectionMatrix();
			self.renderer.setSize(self.WIDTH, self.HEIGHT)
		});
		
	},

	clearScene: function(){
		// remove everythin in the scene
		var scene = this.scene;
		for( var i = scene.children.length - 1; i >= 0; i--) {
			scene.remove(scene.children[i]);
		}
	},

	shapeBy: function(metaKey){
		// groupBy the model and init clouds
		// update shapeKey
		this.shapeKey = metaKey;
		// clear this.clouds
		this.clouds = [];
		this.clearScene();
		// get grouped datasets, each group is going to be a cloud
		var scatterDataSubsets = this.model.groupBy(metaKey);
		var textures = this.textures;

		// make shapeScale for d3.legend
		this.shapeScale = d3.scale.ordinal()
			.domain(Object.keys(scatterDataSubsets))
			.range(_.map(d3.svg.symbolTypes, function(t){
				return d3.svg.symbol().type(t)();
			}));
		
		// symbolTypeScale is used for retrieving a texture instance from textures collection
		var symbolTypeScale = d3.scale.ordinal()
			.domain(Object.keys(scatterDataSubsets))
			.range(textures.pluck('symbolType'));
		
		for (var key in scatterDataSubsets){
			var cloud = new Scatter3dCloud({
				model: scatterDataSubsets[key],
				texture: textures.getTexture(symbolTypeScale(key)), 
				pointSize: this.pointSize,
				sizeAttenuation: this.is3d,
			});

			this.clouds.push(cloud)
			this.scene.add( cloud.points );	
		}

		// re-coloring nodes
		this.colorBy(this.colorKey);
		this.trigger('shapeChanged')
		this.renderScatter();

	},

	renderScatter: function(){
		// update the picking ray with the camera and mouse position
		this.raycaster.setFromCamera( this.mouse, this.camera );

		// calculate objects intersecting the picking ray
		var allPoints = _.map(this.clouds, function(obj){ return obj.points; });
		var intersects = this.raycaster.intersectObjects( allPoints );

		// reset colors
		this.resetColors();

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

		if (this.showStats){
			this.stats.update();	
		}
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

	resetColors: function(){
		// reset colors based on this.metaKey, do not trigger any events.
		for (var i = this.clouds.length - 1; i >= 0; i--) {
			var cloud = this.clouds[i];
			cloud.setColors(this.colorScale, this.colorKey)
		};
	},

	colorBy: function(metaKey){
		// Color points by a certain metaKey
		// update colorKey
		this.colorKey = metaKey;

		var metas = this.model.getAttr(metaKey);
		var uniqueCats = new Set(metas);
		var nUniqueCats = uniqueCats.size;

		var meta = _.findWhere(this.model.metas, {name: metaKey});
		var dtype = meta.type;

		// make colorScale
		if (nUniqueCats < 11){
			var colorScale = d3.scale.category10().domain(uniqueCats);
		} else if (nUniqueCats > 10 && dtype !== 'number') {
			var colorScale = d3.scale.category20().domain(uniqueCats);
		} else {
			var colorExtent = d3.extent(metas);
			var min_score = colorExtent[0],
				max_score = colorExtent[1];
			var colorScale = d3.scale.pow()
				.domain([min_score, (min_score+max_score)/2, max_score])
				.range(["#1f77b4", "#ddd", "#d62728"]);
		}

		this.colorScale = colorScale; // the d3 scale used for coloring nodes

		for (var i = this.clouds.length - 1; i >= 0; i--) {
			var cloud = this.clouds[i];
			cloud.setColors(colorScale, metaKey)
		};
		this.trigger('colorChanged');
		this.renderScatter();
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

