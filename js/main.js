var sd = new ScatterData()

// sd.listenTo(sd, 'sync', function(){
// 	console.log(sd.positions)
// })

var sdv = new Scatter3dView({
	model: sd,
	// WIDTH: $('#canvas-container').width(),
	// HEIGHT: window.innerHeight,
	// container: document.getElementById('canvas-container'),	
})


// var img = new Img({src: 'https://www.google.com/logos/doodles/2017/international-womens-day-2017-5658396607905792-res.png'})
// $('body').append(img.render().el);

// var st = new SymbolTexture();

var legend = new Legend({scatterPlot: sdv})



