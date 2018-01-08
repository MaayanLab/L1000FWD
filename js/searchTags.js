var pert_ids_select = $('#pert_ids').selectize({
	valueField: 'pert_id',
	labelField: 'Name',
	searchField: 'Name',
	sortField: 'Name',
	options: [],
	create:false,
	maxItems: 50,
	render: {
		option: function(item, escape){
			return '<ul>' + 
				'<li>' + escape(item.Name) + '</li>' +
				'<li>pert_id:' + escape(item.pert_id) + '</li>' +
				'</ul>';
		},
		item: function(item, escape){
			return '<div>' + 
				'<span class="drug-name">' + escape(item.Name) + '</span>' + 
				'<span class="drug-pert-id"> (' + escape(item.pert_id) + ')</span>' + 
				'</div>';
		},
	},
	load: function(query, callback){
		if (!query.length) return callback();
		$.ajax({
			url: 'synonyms/' + encodeURIComponent(query),
			type: 'GET',
			dataType: 'json',
			error: function(){
				callback();
			},
			success: function(res){
				return callback(res);
			}
		});
	}
});

var cells_select = $('#cells').selectize({
	valueField: 'value',
	labelField: 'name',
	searchField: 'name',
	sortField: 'name',
	options: [],
	create:false,
	maxItems: 50,
	load: function(query, callback){
		// if (!query.length) return callback();
		$.ajax({
			url: 'cells',
			type: 'GET',
			dataType: 'json',
			error: function(){
				callback();
			},
			success: function(res){
				return callback(res);
			}
		});
	}
});

var times_select = $('#times').selectize({
	valueField: 'value',
	labelField: 'name',
	searchField: 'name',
	sortField: 'name',
	options: [{name: '6H', value: 6}, {name: '24H', value: 24}, {name: '48H', value: 48}],
	create:false,
	maxItems: 3,
});


var postUrl = 'subset';

$('#submit-btn').click(function(e){
	e.preventDefault();

	var pert_ids = $('#pert_ids').val();
	var cells = $('#cells').val();
	var times = $('#times').val();

	if (pert_ids === null){
		alert('Please select at least one drug/compound');
	} else{
		$.ajax(postUrl, {
			contentType : 'application/json',
			type: 'POST',
			data: JSON.stringify({
				pert_ids: pert_ids,
				cells: cells,
				times: times,
			}),
			success: function(result){
				result = JSON.parse(result);
				if (result.error){
					alert(result.error);
				} else{
					if (result.absolute){
						var redirectUrl = result.url
					}else{
						var getUrl = window.location;
						var baseUrl = getUrl .protocol + "//" + getUrl.host + "/" + getUrl.pathname.split('/')[1];
						var redirectUrl = baseUrl + result.url;
					}
					// redirect
					window.location.href = redirectUrl;
				}
			}
		});		
	}
});


function setExample(selectize, example, id_key){
	selectize.clearOptions();
	selectize.addOption(example);
	selectize.setValue(_.pluck(example, id_key));
}

$('.example-btn').click(function(e){
	e.preventDefault();

	var example_drugs = [
		{Name:'TEMSIROLIMUS', pert_id:'BRD-A62025033'}, 
		{Name:'IRINOTECAN', pert_id:'BRD-K08547377'}, 
	];
	var example_cells = [
		{name: 'A375', value: 'A375'}, 
		{name: 'A549', value: 'A549'},
		{name: 'VCAP', value: 'VCAP'},
		{name: 'MCF7', value: 'MCF7'},
		{name: 'PC3', value: 'PC3'},
		{name: 'NPC', value: 'NPC'},
	];
	// var example_times = [{name: '6H', value: 6}, {name: '24H', value: 24}];
	var example_times = [6, 24]

	setExample(pert_ids_select[0].selectize, example_drugs, 'pert_id');
	setExample(cells_select[0].selectize, example_cells, 'value');
	times_select[0].selectize.setValue(example_times,);
	
})


$('#example-btn-2').click(function(e){
	e.preventDefault();

	var example_drugs = [{Name:'FEXOFENADINE', pert_id:'BRD-A73368467'}];
	var example_cells = [{name: 'PC3', value: 'PC3'}];
	// var example_times = [{name: '6H', value: 6}];
	var example_times = [6];

	setExample(pert_ids_select[0].selectize, example_drugs, 'pert_id');
	setExample(cells_select[0].selectize, example_cells, 'value');
	// setExample(times_select[0].selectize, example_times, 'value');
	times_select[0].selectize.setValue(example_times,);
	
})
