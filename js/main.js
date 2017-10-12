var textures = new Textures()


var sd = new ScatterData({
	// n: 10000,
	// url: 'pca'
	// url: 'graph/full'
	// url: 'graph/A375-tSNE_layout.csv'
	// url: 'graph/A375_kNN_5'
	url : url // need to get this variable from server
})

var container = document.getElementById("body")
var width = container.clientWidth;
var height = container.clientHeight;

var sdvDefaultConfig = {
	container: container,
	WIDTH: width,
	HEIGHT: height,	
	model: sd,
	textures: textures,
	// pointSize: 0.1, 
	pointSize: 12,
	is3d: false,
}

sdvConfig = $.extend(sdvDefaultConfig, sdvConfig)

var sdv = new Scatter3dView(sdvConfig)

var legend = new Legend({scatterPlot: sdv, h: window.innerHeight, container: container})

var controler = new Controler({scatterPlot: sdv, h: window.innerHeight, w: 200, container: container})

var search = new SearchSelectize({scatterPlot: sdv, container: "#controls"})

var sigSimSearch = new SigSimSearchForm({scatterPlot: sdv, container: "#controls"})

var overlay = new Overlay({scatterPlot: sdv})
