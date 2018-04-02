var selectizeDom = $("#main-search-box").selectize({

    create: false,
    preload: 'focus',
    options: [],
    optgroups: [
        {id: 'drug', name: 'drugs'},
        {id: 'cell', name: 'cells'},
        {id: 'sig', name: 'signatures'}
    ],
    labelField: 'name',
    valueField: 'id',
    optgroupField: 'type',
    optgroupLabelField: 'name',
    optgroupValueField: 'id',
    optgroupOrder: ['drug', 'cell', 'sig'],
    searchField: ['name', 'id'],
    plugins: ['optgroup_columns'],
    placeholder: 'e.g. Dexamethasone, MCF7',

    render: {
        option: function(item, escape){
            var html = '<ul class="list-unstyled">'
            if (item.type == 'drug') {
                html += '<span class="label label-primary">D</span>'
                html += '<li>' + item.name + '</li>'
                html += '<li>pert_id: ' + item.id + '</li>'
                html += '<li>MOA: ' + item.MOA + '</li>'
                html += '<li>Phase: ' + item.Phase + '</li>'
            } else if (item.type == 'cell') {
                html += '<span class="label label-success">C</span>'
                html += '<li>' + item.id + '</li>'
                html += '<li>#signatures: ' + item.n_sigs + '</li>'
            } else {
                html += '<span class="label label-info">S</span>'
                html += '<li>sig_id: ' + item.id + '</li>'
                html += '<li>drug: ' + item.Perturbation + '</li>'
                html += '<li>cell: ' + item.Cell + '</li>'
                html += '<li>dose: ' + item.Dose + '</li>'
                html += '<li>time: ' + item.Time + '</li>'
                html += '<li>p-value: ' + item['p-value'] + '</li>'
            }
            html += '</ul>'
            return html;
        }
    },
    load: function(query, callback){
        if (!query.length) query = 'vc'; // to preload some options when focused 
        $.ajax({
            url: 'search_all/' + encodeURIComponent(query),
            type: 'GET',
            dataType: 'json',
            error: function(){
                callback();
            },
            success: function(res){
                return callback(res);
            }
        });
    },

    onItemAdd: function (value, $item) {    
        var item = this.options[value];
        var url = '';
        if (item.type == 'drug') {
            url = 'http://amp.pharm.mssm.edu/dmoa/report/' + item.id;
        } else if (item.type == 'cell') {
            url = 'http://amp.pharm.mssm.edu/l1000fwd/graph_page/' + item.name[0];
        } else {
            url = 'http://amp.pharm.mssm.edu/dmoa/sig/' + item.id;
        }
        
        window.location.href = url;
    }

});
