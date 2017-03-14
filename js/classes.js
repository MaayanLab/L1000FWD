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
		texture: null, // the THREE.Texture instance
		data: null, // expect data to be an array of objects
		labelKey: 'sig_id',
		pointSize: 0.01,
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
				// sizeAttenuation: false, 
				map: texture, 
				alphaTest: 0.5, 
				transparent: true,
				opacity: 0.6
				});
	    } else{
			var material = new THREE.PointsMaterial({
				vertexColors: THREE.VertexColors,
				size: 0.1,
				// sizeAttenuation: false, 
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

		this.camera = new THREE.PerspectiveCamera( 70, this.aspectRatio, 0.01, 100 );
		this.camera.position.z = this.pointSize * 120;

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
		// this.raycaster.params.Points.threshold = 0.5;
		this.raycaster.params.Points.threshold = this.pointSize/5;
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
		// this.controls.update();
		// update the picking ray with the camera and mouse position
		this.raycaster.setFromCamera( this.mouse, this.camera );

		// calculate objects intersecting the picking ray
		// var intersects = this.raycaster.intersectObject( this.points );
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

var Legend = Backbone.View.extend({
	// A view for the legends of the Scatter3dView
	// tagName: 'svg',
	defaults: {
		container: document.body,
		scatterPlot: Scatter3dView,
		w: 300,
		h: 800,
	},

	initialize: function(options){
		if (options === undefined) {options = {}}
		_.defaults(options, this.defaults)
		_.defaults(this, options)
		this.setUpDOMs();
		// render if the scatterPlot changed
		this.listenTo(this.scatterPlot, 'shapeChanged', this.render)
		this.listenTo(this.scatterPlot, 'colorChanged', this.render)
	},

	setUpDOMs: function(){
		// set up DOMs for the legends
		this.el = d3.select(this.container)
			.append('svg')
			.attr('width', this.w)
			.attr('height', this.h)
			.style('z-index', 10)
			.style('position', 'absolute')
			.style('left', '0px')
			.style('top', '0px')
			;

		this.g = this.el.append('g')
			.attr('class', 'legend')
			.attr('transform', 'translate(10, 20)');
		this.g.append('g')
			.attr('id', 'legendShape')
			.attr("class", "legendPanel")
			.attr("transform", "translate(0, 0)");
		this.g.append('g')
			.attr('id', 'legendColor')
			.attr("class", "legendPanel")
			.attr("transform", "translate(100, 0)");

	},

	render: function(){
		// set up legend
		// shape legend
		var scatterPlot = this.scatterPlot;
		var legendShape = d3.legend.symbol()
			.scale(scatterPlot.shapeScale)
			.orient("vertical")
			.title(scatterPlot.shapeKey);
		this.g.select("#legendShape")
			.call(legendShape);

		// color legend
		var legendColor = d3.legend.color()
			.title(scatterPlot.colorKey)
			.shapeWidth(20)
			.cells(5)
			.scale(scatterPlot.colorScale);

		this.g.select("#legendColor")
			.call(legendColor);

		return this;
	},

});



var Controler = Backbone.View.extend({

	defaults: {
		container: document.body,
		scatterPlot: Scatter3dView,
		w: 300,
		h: 800,
	},

	initialize: function(options){
		if (options === undefined) {options = {}}
		_.defaults(options, this.defaults)
		_.defaults(this, options)

		this.model = this.scatterPlot.model;

		this.listenTo(this.model, 'sync', this.render);

		var scatterPlot = this.scatterPlot;

		this.listenTo(scatterPlot, 'shapeChanged', this.changeSelection)

		scatterPlot.listenTo(this, 'shapeChanged', function(selectedMetaKey){
			scatterPlot.shapeBy(selectedMetaKey);
		});
		scatterPlot.listenTo(this, 'colorChanged', function(selectedMetaKey){
			scatterPlot.colorBy(selectedMetaKey);
		});

	},

	render: function(){
		// set up DOMs for the controler
		this.el = d3.select(this.container)
			.append('div')
			.attr('id', 'controls')
			.style('width', this.w)
			.style('height', this.h)
			.style('z-index', 10)
			.style('position', 'absolute')
			.style('right', '0px')
			.style('top', '0px');

		var model = this.model;
		// filter out metas used as index
		var metas = _.filter(model.metas, function(meta){ return meta.nUnique < model.n; });
		var self = this;


		// Shapes: 
		var shapeControl = this.el.append('div')
			.attr('class', 'form-group');
		shapeControl.append('label')
			.attr('class', 'control-label')
			.text('Shape by:');

		var shapeSelect = shapeControl.append('select')
			.attr('id', 'shape')
			.attr('class', 'form-control')
			.on('change', function(){
				var selectedMetaKey = d3.select('#shape').property('value');
				self.trigger('shapeChanged', selectedMetaKey)
			});

		var shapeOptions = shapeSelect
			.selectAll('option')
			.data(_.pluck(metas, 'name')).enter()
			.append('option')
			.text(function(d){return d;})
			.attr('value', function(d){return d;});

		// Colors
		var colorControl = this.el.append('div')
			.attr('class', 'form-group')
		colorControl.append('label')
			.attr('class', 'control-label')
			.text('Color by:');

		var colorSelect = colorControl.append('select')
			.attr('id', 'color')
			.attr('class', 'form-control')
			.on('change', function(){
				var selectedMetaKey = d3.select('#color').property('value');
				self.trigger('colorChanged', selectedMetaKey)
			});

		var colorOptions = colorSelect
			.selectAll('option')
			.data(_.pluck(metas, 'name')).enter()
			.append('option')
			.text(function(d){return d;})
			.attr('value', function(d){return d;});

		return this;
	},

	changeSelection: function(){
		// change the current selected option to value
		$('#shape').val(this.scatterPlot.shapeKey); 
		$('#color').val(this.scatterPlot.colorKey);
	},

});

