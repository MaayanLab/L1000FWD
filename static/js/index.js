// Search bar
var selectizeDom = $("#main-search-box").selectize({

    create: false,
    // preload: 'focus',
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
                html += '<span class="badge badge-primary">D</span>'
                html += '<li>' + item.name + '</li>'
                html += '<li>pert_id: ' + item.id + '</li>'
                html += '<li>MOA: ' + item.MOA + '</li>'
                html += '<li>Phase: ' + item.Phase + '</li>'
            } else if (item.type == 'cell') {
                html += '<span class="badge badge-success">C</span>'
                html += '<li>' + item.id + '</li>'
                html += '<li>#signatures: ' + item.n_sigs + '</li>'
            } else {
                html += '<span class="badge badge-info">S</span>'
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
        // if (!query.length) query = 'vc'; // to preload some options when focused 
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
            url = 'graph_page/' + item.name[0];
        } else {
            url = 'http://amp.pharm.mssm.edu/dmoa/sig/' + item.id;
        }
        
        location.href = url;
    }

});


// highcharts
// var colors = Highcharts.getOptions().colors;
var colors = d3.scale.category20(),
    colors10 = d3.scale.category10();
// get the data for the stats
var cat_inner = 'Cell',
    cat_outer = 'Phase';

$.getJSON('stats/'+ cat_inner+'/' + cat_outer, function(data){
    var categories = _.map(data, function(rec){
        return rec.drilldown.name;
    });
    var subCategories = data[0].drilldown.categories;
    // Make unknown to be gray 
    if (subCategories.indexOf('unknown') !== -1) {
        var idx = subCategories.indexOf('unknown');
        greyIdx = 7;
        var colorsSub = colors10
        if (subCategories.length == 20){
            var greyIdx = 15;
            colorsSub = colors;
        }
        var elem = subCategories[greyIdx];
        subCategories[greyIdx] = 'unknown';
        subCategories[idx] = elem;
    };


    // include colors into the records
    for (var i = data.length - 1; i >= 0; i--) {
        var rec = data[i];
        rec['color'] = colors(i);
        if (rec['drilldown']['name'] != 'other'){
            rec['url'] = 'graph_page/' + rec['drilldown']['name'] + '_kNN_5'
        }
        
    }

    var browserData = [],
    versionsData = [],
    i,
    j,
    dataLen = data.length,
    drillDataLen,
    brightness;
// Build the data arrays
for (i = 0; i < dataLen; i += 1) {

    // add browser data
    browserData.push({
        name: categories[i],
        y: parseFloat(data[i].y.toFixed(2)),
        color: data[i].color,
        // add url for cells
        url: data[i].url
    });

    // add version data
    drillDataLen = data[i].drilldown.data.length;
    for (j = 0; j < drillDataLen; j += 1) {
        brightness = 0.2 - (j / drillDataLen) / 5;
        var subCategory = data[i].drilldown.categories[j];
        var idx = subCategories.indexOf(subCategory);
        versionsData.push({
            name: subCategory,
            y: parseFloat(data[i].drilldown.data[j].toFixed(2)),
            // color: Highcharts.Color(data[i].color).brighten(brightness).get()
            color: colorsSub(idx)
        });
    }
}

// Create the chart
Highcharts.chart('highcharts-container', {
    chart: {
        type: 'pie'
    },
    title: {
        text: ''
    },
    subtitle: {
        text: 'Click cell line slice to launch cell-specific visualization'
    },
    yAxis: {
        title: {
            text: ''
        }
    },
    plotOptions: {
        pie: {
            shadow: false,
            center: ['50%', '50%']
        }
    },
    tooltip: {
        formatter: function(){
            return '<b>' + this.series.name + ': ' + this.point.name + '</b><br>' + 
                this.y + '%';
        }
    },
    series: [{
        name: cat_inner,
        data: browserData,
        size: '60%',
        cursor: 'pointer',
        point: {
            events: {
                click: function(){
                    if (this.options.url != undefined){
                        location.href = this.options.url;    
                    }
                }
            }
        },
        dataLabels: {
            formatter: function () {
                return this.y > 5 ? this.point.name : null;
            },
            color: '#ffffff',
            distance: -30,
            defer: true,
            enabled: true
        }
    }, {
        name: cat_outer,
        data: versionsData,
        size: '80%',
        innerSize: '60%',
        dataLabels: {
            formatter: function () {
                // display only if larger than 1
                return this.y > 1 ? '<b>' + this.point.name + ':</b> ' +
                    this.y + '%' : null;
            },
            defer: true,
            enabled: true            
        },
        id: 'versions'
    }],
    responsive: {
        rules: [{
            condition: {
                maxWidth: 400
            },
            chartOptions: {
                series: [{
                    id: 'versions',
                    dataLabels: {
                        enabled: false
                    }
                }]
            }
        }]
    },
    credits: {
        enabled: false
    },

});


})
