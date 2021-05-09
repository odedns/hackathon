var ctx = document.getElementById("values-chart");
var xStart = new Date().getMilliseconds();
var mydata = [];

var plen = getParameterByName("plen") || 100
var qlen = getParameterByName("qlen") || 75;
var limit = getParameterByName("limit") || 2000;
var delta = getParameterByName("delta") || 100;
var colName = getParameterByName("colName") || "materna1";
console.log("plen = "+  plen + " qlen = " + qlen + " limit = " + limit + " delta =" + delta + " colName=" + colName);


getData();
var valChart = new Chart(ctx, {
    type: 'line',
    data: {
      datasets: [{
            data: mydata, 
            label: "CPU",
            borderColor: 'blue',
            borderWidth: 1,
            fill: false
        },{
            data: mydata, 
            label: "Anomality",
            borderColor: 'red',
            borderWidth: 8,
            fill: false
        }

      ]
    },
    options: {
      title: {
        display: true,
        text: 'CPU values'
      },
      responsive: true,
        maintainAspectRatio: true,
        spanGaps: false,
      scales: {
        x: {
            type: 'timeseries',
            position: 'bottom',
            display: true,
            /*
            time: {
                stepSize: 10
            },
            */
            ticks: {
                source: 'data'
            }
        }
       
        
    }
    }
  });

  var ctx2 = document.getElementById("score-chart");
  var scoreChart = new Chart(ctx2, {
    type: 'line',
    data: {
      datasets: [{
            data: mydata, 
            label: "Score",
            borderColor: 'blue',
            fill: false
        },
        {
            data: mydata, 
            label: "Anomality",
            borderColor: 'red',
            borderWidth: 10,
            fill: false
        }
      ]
    },
    options: {
      title: {
        display: true,
        text: 'CPU Anomality score'
      },
      responsive: true,
        maintainAspectRatio: true,
        spanGaps: false,
     

      scales: {
        x: {
            type: 'timeseries',
            position: 'bottom',
            display: true,
            /*
            time: {
                stepSize: 10
            },
            */
            ticks: {
                source: 'data'
            }
        }
       
        
    }
    }
  });


  setInterval(update,10000);

  function update() {
      console.log("in update()");
      let url = "/s2g?delta="+delta + "&plen="+plen+"&qlen="+ qlen+ "&colName="+colName+"&limit="+ limit;
      console.log("url=" + url);
      $.getJSON( url, function( data ) {
        valChart.data.datasets[0].data = data.values;
        valChart.data.datasets[1].data = data.anom_val;

        valChart.update();
        scoreChart.data.datasets[0].data = data.scores;
        scoreChart.data.datasets[1].data = data.anom_score;

        scoreChart.update();
        delta+=100;
        console.log("updated chart new delta=" + delta);

    });
  }

  

  function getData() {
    console.log("in get data");
    let url = "/s2g?delta=0" + "&plen="+plen+"&qlen="+ qlen+ "&colName="+colName+"&limit="+ limit;
    console.log("url=" + url);
    $.getJSON( url, function( data ) {
        valChart.data.datasets[0].data = data.values;
        valChart.data.datasets[1].data = data.anom_val;

        valChart.update();
        scoreChart.data.datasets[0].data = data.scores;
        scoreChart.data.datasets[1].data = data.anom_score;

        scoreChart.update();
        console.log("updated chart");

    });
  }
  function getParameterByName(name) {
    var match = RegExp('[?&]' + name + '=([^&]*)').exec(window.location.search);
    return match && decodeURIComponent(match[1].replace(/\+/g, ' '));
  }  
