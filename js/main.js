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

var sdv = new Scatter3dView({
	model: sd,
	textures: textures,
	// pointSize: 0.1, 
	pointSize: 12,
	is3d: false,
	colorKey: 'cell',
	shapeKey: 'time',
	labelKey: ['sig_id', 'perturbation'],
})

var legend = new Legend({scatterPlot: sdv, h: window.innerHeight})

var controler = new Controler({scatterPlot: sdv, h: window.innerHeight, w: 200})

var search = new SearchSelectize({scatterPlot: sdv, container: "#controls"})

var sigSimSearch = new SigSimSearchForm({scatterPlot: sdv, container: "#controls"})

var overlay = new Overlay({scatterPlot: sdv})
