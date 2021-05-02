var ctx = document.getElementById("values-chart");
var xStart = new Date().getMilliseconds();
var mydata = [];
getData();
var valChart = new Chart(ctx, {
    type: 'line',
    data: {
      datasets: [{
            data: mydata, 
            label: "CPU",
            borderColor: 'blue',
            borderWidth: 2,
            fill: false
        },{
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
        text: 'CPU values',
        responsive: true,
        maintainAspectRatio: false,
        spanGaps: true

      },

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
        text: 'CPU Anomality score',
        responsive: true,
        maintainAspectRatio: false,
        spanGaps: true
      },

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


  function update() {
      console.log("in update()");
      $.getJSON( "/data?limit=10?skip=50", function( data ) {
        //console.log(data);
        console.log("x = " + data[0].x);
        
        chart.data.datasets[0].data.splice(0,10); 
        chart.data.datasets[0].data.push(...data);
        chart.update();
        console.log("updated chart");

    });
  }

  

  function getData() {
    console.log("in get data");
    $.getJSON( "/s2g?limit=5000", function( data ) {
        valChart.data.datasets[0].data = data.values;
        valChart.data.datasets[1].data = data.anom_val;

        valChart.update();
        scoreChart.data.datasets[0].data = data.scores;
        scoreChart.data.datasets[1].data = data.anom_score;

        scoreChart.update();
        console.log("updated chart");

    });
  }