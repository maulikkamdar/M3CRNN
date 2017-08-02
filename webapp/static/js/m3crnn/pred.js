function select_data(pred_type) {
  jQuery("#curtype").html("<b>" + pred_type.toUpperCase()  + "</b>");
  jQuery.get("/m3crnn/get_pred_list?q=" + pred_type, function(data) {
    jQuery("#ptall").html("");
    for (k in data["data"]) {
      jQuery(".list-group").append('<button type="button" class="list-group-item" onclick="javascript:openscan(\'' + data["data"][k]["shahash"] + '\', \''+pred_type+'\')">' + data["data"][k]["shahash"].substr(0, 30) + "..." + '</button>')
    }
  }, 'json');
}

function openscan(scan, data_type) {
  jQuery.get("/m3crnn/get_pred_scan?shahash=" + scan + "&q=" + data_type, function(data) {
    jQuery("#mriscan").html("");
    jQuery("#shahash").val(scan)
    jQuery("#scanid").html("<b>" + scan.toUpperCase()  + "</b>");
    jQuery("#datatype").val(data_type)
    jQuery("#mriinfo").html("<h5>" + data["patient_id"] + ", Slice Count: " + data["slice_count"] + "</h5>")
    jQuery("#mriscan").html(data["html"])
  }, 'json');
}

console.log('here');
select_data("tp");