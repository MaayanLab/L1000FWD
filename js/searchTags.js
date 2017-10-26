$('#pert_ids').selectize({
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
		}
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

$('#cells').selectize({
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

$('#times').selectize({
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
				var getUrl = window.location;
				var baseUrl = getUrl .protocol + "//" + getUrl.host + "/" + getUrl.pathname.split('/')[1];
				// redirect
				window.location.href = baseUrl + result.url;
			}
		});		
	}


})
