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
	url: 'pca'
})

var sdv = new Scatter3dView({
	model: sd,
	textures: textures,
	// pointSize: 0.1, 
	pointSize: 10,
	is3d: false,
})

var legend = new Legend({scatterPlot: sdv})
// legend.listenTo(sdv, 'colorChanged', legend.render)

var controler = new Controler({scatterPlot: sdv})

var overlay = new Overlay({scatterPlot: sdv})
