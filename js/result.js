var textures = new Textures()


var sd = new ScatterData({
	url: 'result/graph/' + result_id
})

var sdv = new Scatter3dView({
	model: sd,
	textures: textures,
	// pointSize: 0.1, 
	pointSize: 12,
	is3d: false,
	colorKey: 'scores',
	shapeKey: 'time',
	labelKey: ['sig_id', 'perturbation'],
})

var legend = new Legend({scatterPlot: sdv, h: window.innerHeight})

var controler = new Controler({scatterPlot: sdv, h: window.innerHeight, w: 200})

var search = new SearchSelectize({scatterPlot: sdv, container: "#controls"})

var sigSimSearch = new SigSimSearchForm({scatterPlot: sdv, container: "#controls", result_id: result_id})

var resultModalBtn = new ResultModalBtn({scatterPlot: sdv, container: "#controls", result_id: result_id})

var resultModal = new ResultModal({scatterPlot: sdv});

var overlay = new Overlay({scatterPlot: sdv})
