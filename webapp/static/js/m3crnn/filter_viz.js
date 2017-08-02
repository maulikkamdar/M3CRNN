if (!storage.getItem("user")) {
  jQuery.getJSON("http://jsonip.com/?callback=?", function (data) {
      console.log(data)
      jQuery.ajax({
        url: "/create_user?ip=" + data["ip"],
        type: 'GET',
        beforeSend: function(){
            jQuery('.splashScreenExplorer').show();
        },
        complete: function(){
            jQuery('.splashScreenExplorer').hide();
        },
        success: function(response) {     
            output = JSON.parse(response);
            storage.setItem("user", output["user_id"]);
        },
        error: function(error) {
            console.log(error);
        }
    });
  });
} 


function select_data(data_type) {
  jQuery("#curtype").html("<b>" + data_type.toUpperCase()  + "</b>");
  jQuery.get("/m3crnn/get_patient_list?q=" + data_type, function(data) {
    jQuery("#ptall").html("");
    for (k in data["data"]) {
      if (data["data"][k]["done"] == -1) {
        span_class = "glyphicon glyphicon-pencil"
      } else if (data["data"][k]["done"] == 0) {
        span_class = "glyphicon glyphicon-remove"
      } else {
        span_class = "glyphicon glyphicon-ok"
      }
      jQuery(".list-group").append('<button type="button" class="list-group-item" onclick="javascript:runscan(\'' + data["data"][k]["shahash"] + '\', \''+data_type+'\')">' + data["data"][k]["shahash"].substr(0, 30) + "..." + 
        '<span class="' + span_class + ' pull-right" aria-hidden="true" id="' + data["data"][k]["shahash"] + '_icon"></span></button>')
    }
  }, 'json');
}

function runscan(scan, data_type) {
  jQuery.get("/m3crnn/run_scan?shahash=" + scan + "&q=" + data_type + "&user_id=" + storage.getItem("user"), function(data) {
    jQuery("#scanid").html("<b>" + scan.toUpperCase()  + "</b>");
    jQuery("#mriinfo").html("<h5>" + data["patient_id"] + ", Slice Count: " + data["slice_count"] + ", Methylation State: " + data["meth_state"] + "</h5>")
    jQuery("#prediction").html(data["preds"])
    jQuery("#probability").html(data["scores"])
    jQuery("#actual").html(data["meth_state"])
  }, 'json');
}

function launch(filter_id) {
  jQuery.get("/m3crnn/visualize_output?filter_info=" + filter_id + "&user_id=" + storage.getItem("user"), function(data) {
    fparts = filter_id.split("_")
    ftype = fparts[2] == "foviz" ? "Filter Output" : "RELU Output"
    filter_name = fparts[0].substr(0,3) + " Layer " + fparts[0].substr(3,) + " Filter " + fparts[1] + " " + ftype
    jQuery("body").append("<div class='overlay_viz' id='viz_" + filter_id + "'><h4>" + filter_name + " <span class='pull-right'><a href='javascript:show_hide_div(\"viz_" + filter_id + "\")'><span class='glyphicon glyphicon-remove' aria-hidden='true'></span></a></span></h4><hr></div>")
    jQuery("#viz_"+filter_id).draggable()
    jQuery("#viz_"+filter_id).append(data["html"]);
  }, 'json');
}

function launch_mri() {
  jQuery.get("/m3crnn/visualize_mri?user_id=" + storage.getItem("user"), function(data) {
    jQuery("body").append("<div class='overlay_viz' id='viz_mri'><h4> Original MRI scan <span class='pull-right'><a href='javascript:show_hide_div(\"viz_mri\")'><span class='glyphicon glyphicon-remove' aria-hidden='true'></span></a></span></h4><hr></div>")
    jQuery("#viz_mri").draggable()
    jQuery("#viz_mri").append(data["html"]);
  }, 'json');
}

function show_hide_div(div_id) {
  $("#" + div_id).remove();
}

select_data("train");

model_configs = {"CNN": {"CNN1": 8, "CNN2": 8, "CNN3": 8, "CNN4": 8}}

for (k in model_configs["CNN"]) {
  //<button type="button" class="btn btn-primary" onclick="javascript:save(\'' + k + '_' + m + '_fviz\')" title="Visualize Filter"><span class="glyphicon glyphicon-filter" aria-hidden="true" id="' + k + '_' + m + '_fviz"></span></button>
  for (m = 0; m < model_configs["CNN"][k]; m++) {
    jQuery("#" + k + "Viz").append('<div class="row-fluid"><div class="col-md-3"><h4> F' + (m+1) + ': </h4></div><div class="col-md-9"><button type="button" class="btn btn-primary" onclick="javascript:launch(\'' + k + '_' + m + '_foviz\')" title="Visualize Filter Output"><span class="glyphicon glyphicon-eye-open" aria-hidden="true" id="' + k + '_' + m + '_foviz"></span></button><button type="button" class="btn btn-primary" onclick="javascript:launch(\'' + k + '_' + m + '_relu\')" title="Visualize RELU Output"><span class="glyphicon glyphicon-eye-close" aria-hidden="true" id="' + k + '_' + m + '_relu"></span></button><hr></div></div>')
  }
}
