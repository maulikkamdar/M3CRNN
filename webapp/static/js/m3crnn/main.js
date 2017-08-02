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
      jQuery(".list-group").append('<button type="button" class="list-group-item" onclick="javascript:openscan(\'' + data["data"][k]["shahash"] + '\', \''+data_type+'\')">' + data["data"][k]["shahash"].substr(0, 30) + "..." + 
        '<span class="' + span_class + ' pull-right" aria-hidden="true" id="' + data["data"][k]["shahash"] + '_icon"></span></button>')
    }
  }, 'json');
}

function openscan(scan, data_type) {
  jQuery.get("/m3crnn/get_patient_scan?shahash=" + scan + "&q=" + data_type, function(data) {
    jQuery("#mriscan").html("");
    jQuery("#seqend").val(data["seq_end"])
    jQuery("#tumorend").val(data["tumor_end"])
    jQuery("#seqstart").val(data["seq_start"])
    jQuery("#tumorstart").val(data["tumor_start"])
    jQuery("#shahash").val(scan)
    jQuery("#scanid").html("<b>" + scan.toUpperCase()  + "</b>");
    jQuery("#datatype").val(data_type)
    jQuery("#mriinfo").html("<h5>" + data["patient_id"] + ", Slice Count: " + data["slice_count"] + ", Methylation State: " + data["meth_state"] + "</h5>")
    jQuery("#mriscan").html(data["html"])
  }, 'json');
}

function make_decision(decision) {
  seqend = jQuery("#seqend").val()
  seqstart = jQuery("#seqstart").val()
  tumorend = jQuery("#tumorend").val()
  tumorstart = jQuery("#tumorstart").val()
  shahash = jQuery("#shahash").val()
  q = jQuery("#datatype").val()
  query = "/m3crnn/make_decision?q=" + q + "&shahash=" + shahash + 
          "&seq_start=" + seqstart + "&seq_end=" + seqend + 
          "&tumor_start=" + tumorstart + "&tumor_end=" + tumorend + "&is_valid=" + decision
  jQuery.get(query, function(data) {
    if (data["output"] == "Success") {
      if (decision == 'true') {$("#" + shahash + "_icon").attr('class', 'glyphicon glyphicon-ok pull-right');}
      if (decision == 'false') {$("#" + shahash + "_icon").attr('class', 'glyphicon glyphicon-remove pull-right');}
    }
  }, 'json');
}

function save(){
  jQuery.get("/m3crnn/save_annotations", function(data) {
    console.log(data["output"])
  }, 'json'); 
}


select_data("train");