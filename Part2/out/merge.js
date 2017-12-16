var fs = require('fs');
var print = console.log;

var files = [
	"./knn.json",
	"./mf_hybrid.json"
];

var ret = [];
files.forEach(function(file, i) {
	var data = require(file);
	print(`Length of ${i} => ${data.length}`)
	ret = ret.concat(data);
	print(`Length of merged => ${ret.length}`)
});

var json = JSON.stringify(ret);
fs.writeFileSync('./eval.json', json);