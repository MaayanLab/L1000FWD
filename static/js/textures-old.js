var Img = Backbone.View.extend({
	// tagName: 'img',
	attributes: {
		width: 32,
		height: 32,
		src: null,
	},

	template: _.template('<img width="<%= width %>" height=""<%= height %>"" src="<%= src %>">'),

    initialize: function(options){

    	if (options === undefined) {options = {}}
		this.attributes.src = options.src;

    	this.render();
    },

    render: function(){

    	this.$el.html(this.template(this.attributes))
    	// this.el = this.template(this.attributes);
    	this.$('img').on('load', _.bind(this.onLoad, this));
    	return this;
    },

    onLoad: function() {
        // Callback when img is loaded
        console.log('img loaded')
        this.trigger('imgLoaded', this.$('img')[0]);
        $('body').html(this.el)
    },
});



var SymbolTexture = Backbone.View.extend({
	// view for THREE.Texture from a 2d canvas from d3.svg.symbol
	// tagName: 'canvas',

	defaults: {
		symbolType: 'triangle-up',
	},
	
	events: {
		// load: imgLoaded,
	},

	initialize: function(options){
		if (options === undefined) {options = {}}
		_.defaults(options, this.defaults)
		_.defaults(this, options)

		var doctype = '<?xml version="1.0" standalone="no"?>' 
		 + '<!DOCTYPE svg PUBLIC "-//W3C//DTD SVG 1.1//EN" "http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd">';

		var svg = document.createElement('svg');
		this.svg = d3.select(svg)
			.attr('width', 32)
			.attr('height', 32);

		// draw symbol on the svg
		var arc = d3.svg.symbol().type(this.symbolType)
			.size(320);

		var line = this.svg.append('path')
			.attr('d', arc)
			.attr('fill', 'green')
			.attr('stroke', 'red')
			.attr('transform', 'translate(16, 16)');

		// serialize our SVG XML to a string.
		var source = (new XMLSerializer()).serializeToString(this.svg.node());

		// create a file blob of our SVG.
		var blob = new Blob([ doctype + source], { type: 'image/svg+xml;charset=utf-8' });

		var url = window.URL.createObjectURL(blob);

		// Put the svg into an image tag so that the Canvas element can read it in.
		this.img = new Img({src: url});
		console.log(url)
		console.log(this.img)
		console.log(this.img.$('img')[0]);
		this.listenTo(this.img, 'imgLoaded', function(data){
			this.render(data)
		});

	},

	render: function(img){
		// Now that the image has loaded, put the image into a canvas element.
		// var canvas = d3.select('body').append('canvas').node();
		console.log('SymbolTexture.render called')
		var canvas = document.createElement('canvas');
		canvas.width = 32;
		canvas.height = 32;
		var ctx = canvas.getContext('2d');
		ctx.drawImage(img, 0, 0);
		var canvasUrl = canvas.toDataURL("image/png");
		console.log(canvasUrl)

		this.canvas = canvas;
	},

});
