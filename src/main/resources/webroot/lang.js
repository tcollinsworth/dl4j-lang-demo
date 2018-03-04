$(document).ready(function(){

  function alertFailure(jqxhr, textStatus, error) {
    alert("Error retry\r\n" + JSON.stringify(jqxhr, null, '  '))
  }

  $('#reset').click(function() {
    postData(Config.getModelAdminUrl(), {resetModel: true},
      function(data, textStatus, jqxhr) {
        alert("Successfully reset model and cleared training and eval DataSets")
      },
      function(jqxhr, textStatus, error) {
        alert("Error resetting model\r\n" +  JSON.stringify(jqxhr, null, '  '))
      })
  })

  $('#save').click(function() {
    postData(Config.getModelAdminUrl(), {saveModel: true, modelFilename: $('#filename').val()},
      function(data, textStatus, jqxhr) {
        alert("Successfully saved model " + $('#filename').val())
      },
      function(jqxhr, textStatus, error) {
        alert("Error saving model\r\n" +  JSON.stringify(jqxhr, null, '  '))
      })
  })

  $('#load').click(function() {
    postData(Config.getModelAdminUrl(), {loadModel: true, modelFilename: $('#filename').val()},
      function(data, textStatus, jqxhr) {
        alert("Successfully loaded model " + $('#filename').val())
      }, function(jqxhr, textStatus, error) {
        alert("Error loading model\r\n" +  JSON.stringify(jqxhr, null, '  '))
      })
  })

  function postData(url, data, successCB, failureCB) {
    $.ajax({
    	url,
    	type: 'POST',
    	data: JSON.stringify(data),
    	contentType: 'application/json; charset=utf-8',
    	dataType: 'json'
    })
    .done(successCB)
    .fail(failureCB);
  }

})
