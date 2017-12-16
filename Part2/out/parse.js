var _ = require('underscore');
var print = console.log;
var fs = require('fs');
var Baby = require('babyparse');


var json =  require('./eval.json');

// build knns
var knn = _.where(json, {name: 'knn'})
var knn_groups = _.groupBy(knn, 'sample')
// print(knn_groups)
var knn_vs_mae = {}
Object.keys(knn_groups).forEach(function(key) {
	print(key);
	var evals = knn_groups[key];

	var errors = evals.map(e => e.test.mae)
	var k_s = evals.map(e => e.k)

	knn_vs_mae[key] = [k_s, errors];
});

var _json = JSON.stringify(knn_vs_mae);
fs.writeFileSync('./graphs/knnVsMae.json', _json)


// build knns
var mf = _.where(json, {name: 'mf'})
var mf_groups = _.groupBy(mf, 'sample');
var mf_vs_mae = {}
Object.keys(mf_groups).forEach(function(key) {
	var evals = mf_groups[key];

	print(evals)
	var errors = evals.map(e => e.test.mae)
	var f_s = evals.map(e => e.f)

	//print(errors);
	//print(k_s)

	mf_vs_mae[key] = [f_s, errors];
});

var _json = JSON.stringify(mf_vs_mae);
fs.writeFileSync('./graphs/mfsVsMae.json', _json);

// build bar chart dataframe for time
// Looks something like
// number of users, model, time
// then should be graphable like 
// df.pivot(index='m_items', columns='model', values='time').plot(kind='bar')
// just take for the highest_k

var evals = json.filter(function(eval) {
	return (eval.name === 'hybrid' || eval.name === 'knn' || eval.name === 'mf') &&
		(eval.k === 40 || eval.f === 40)
});

var timeData = [];
evals.forEach(function(eval, i) {
	timeData.push({
		m_items: eval.sample[1],
		model: eval.name,
		time: eval.time
	});
});

evals = json.filter(function(eval) {
	return (eval.name == 'knn' || eval.name === 'mf' || eval.name === 'hybrid') &&
		(eval.k === 30 || eval.f === 30);
});

var maeData = [];
evals.forEach(function(eval, i) {
	maeData.push({
		m_items: eval.sample[1],
		model: eval.name,
		mae: eval.test.mae	
	})
});

var csv = Baby.unparse(timeData);
fs.writeFileSync('./graphs/item_size_VS_time.csv', csv);

csv = Baby.unparse(maeData);
fs.writeFileSync('./graphs/item_size_vs_MAE.csv', csv);







