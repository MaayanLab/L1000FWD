var textures = new Textures()


var sd = new ScatterData({
	url: 'result/graph/' + result_id
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
	colorKey: 'scores',
	shapeKey: 'Time',
	labelKey: ['Batch', 'Perturbation', 'Cell', 'Dose', 'Time'],
})

var legend = new Legend({scatterPlot: sdv, h: window.innerHeight, container: container})

var controler = new Controler({scatterPlot: sdv, h: window.innerHeight, w: 200, container: container})

var search = new SearchSelectize({scatterPlot: sdv, container: "#controls"})

var sigSimSearch = new SigSimSearchForm({scatterPlot: sdv, container: "#controls", result_id: result_id})

var resultModalBtn = new ResultModalBtn({scatterPlot: sdv, container: document.body, result_id: result_id, container: container})

var resultModal = new ResultModal({scatterPlot: sdv});

var overlay = new Overlay({scatterPlot: sdv})
