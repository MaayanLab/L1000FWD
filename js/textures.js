/*
Classes for the textures.
*/
var Texture = Backbone.Model.extend({
	defaults: {
			symbolType: 'diamond',
		},

	url: function(){
		return 'lib/textures/d3-symbols/' +this.symbolType + '.png';
	},

	initialize: function(options){
		// called on construction
		if (options === undefined) {options = {}}
		_.defaults(options, this.defaults)
		_.defaults(this, options)
		
		var loader = new THREE.TextureLoader()
		var self = this;
		loader.load(this.url(), function(texture){
			self.texture = texture;
			self.trigger('sync');
		});
	},
});

var Textures = Backbone.Collection.extend({
	model: Texture,
	symbolTypes: d3.svg.symbolTypes,
	initialize: function(){
		this.synced = 0;
		var symbolTypes = this.symbolTypes;
		var self = this;

		// add all the textures
		for (var i = 0; i < symbolTypes.length; i++) {	
			var model = new Texture({symbolType: symbolTypes[i]})
			this.add(model);

			this.listenTo(model, 'sync', function(){
				self.synced ++
				self.trigger('nLoaded', self.synced)
			})
		};


		this.listenTo(this, 'nLoaded', function(n){
			console.log('Textures collection nLoaded:', n)
			if (n === symbolTypes.length){
				console.log('all textures loaded')
				self.trigger('allLoaded');
			}
		});
	},

	getTexture: function(symbolType){
		// get the THREE.Texture instance
		var texture = this.findWhere({symbolType: symbolType})
		if (texture === undefined){
			return null;
		} else{
			return texture.texture;
		}
	}

});


