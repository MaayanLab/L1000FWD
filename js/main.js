var textures = new Textures()

// textures.listenTo(textures, 'allLoaded', function(){
// 	// load the textures collection first
// 	var sd = new ScatterData()

// 	var sdv = new Scatter3dView({
// 		model: sd,
// 		// WIDTH: $('#canvas-container').width(),
// 		// HEIGHT: window.innerHeight,
// 		// container: document.getElementById('canvas-container'),	
// 		textures: textures,
// 	})

// })


var sd = new ScatterData({
	// n: 10000,
	// url: 'pca'
	url: 'graph'
})

var container = document.getElementById("body")
var width = container.clientWidth;
var height = container.clientHeight;

var sdv = new Scatter3dView({
	container: container,
	WIDTH: width,
	HEIGHT: height,	
	model: sd,
	textures: textures,
	// pointSize: 0.1, 
	pointSize: 12,
	is3d: false,
	colorKey: 'Cell',
	shapeKey: 'Time',
	labelKey: ['Batch', 'Perturbation', 'Cell', 'Dose', 'Time', 'Phase', 'MOA'],
})

var legend = new Legend({scatterPlot: sdv, h: window.innerHeight, container: container})

var controler = new Controler({scatterPlot: sdv, h: window.innerHeight, w: 200, container: container})

var search = new SearchSelectize({scatterPlot: sdv, container: "#controls"})

var sigSimSearch = new SigSimSearchForm({scatterPlot: sdv, container: "#controls"})

var overlay = new Overlay({scatterPlot: sdv})
