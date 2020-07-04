let sendText = document.getElementById('sendText');

  sendText.onclick = function(element) {
    // chrome.tabs.query({active: true, currentWindow: true}, function(tabs) {
	// 	chrome.tabs.executeScript(
	// 		tabs[0].id,
	// 		{code: 'var div=document.createElement("div"); document.body.prepend(div); div.innerText="hey"; div.id="modal1"; div.style.backgroundColor="' + color + '"'
	// 	});
	// });
	
	var resp;
	chrome.tabs.query({active: true, currentWindow: true},function(tabs) {
		chrome.tabs.executeScript(
			null,
			{code: "window.getSelection().toString()"}, async function(selection){
				var xmlhttp = new XMLHttpRequest();   // new HttpRequest instance 
				var theUrl = "http://localhost:5000/get_prediction";
				xmlhttp.open("POST", theUrl);
				xmlhttp.setRequestHeader("Content-Type", "application/json;charset=UTF-8");
				xmlhttp.send(JSON.stringify({ "text": selection[0] }));
				xmlhttp.onreadystatechange =  async function() {
					if (this.readyState == 4) {
					  //JSON.parse does not evaluate the attacker's scripts.
					  console.log(this.responseText);
					  resp =  JSON.parse(this.responseText);
					}
				}
			}
		);
	});

	var modal = 
				'<div id="modal-content" class="modal">' +
					'<div class="modal-header">' +
						'<span id="cross" class="close" onclick="document.getElementById(`modal-content`).style.display = `none`;">&times;</span>' +
						'<h2>'+resp+'</h2>' +
					'</div>' +

					'<div class="modal-body">' +
	  					'<p>Some text in the Modal..</p>' +
					'</div>' +
				'</div>'
				
	var modal_style = 
				'.modal-header {' +
					'padding: 0px 0px 16px 16px;' +
					'background-color: #5cb85c;' +
					'color: white;' +
					'}' +
				
				'.modal-body {padding: 2px 16px;}' +
				
				'.modal {' +
					'position: relative;' +
					'background-color: #fefefe;' +
					'margin: auto;' +
					'padding: 0px;' +
					'border: 1px solid #888;' +
					'width: 50%;' +
					'box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2),0 6px 20px 0 rgba(0,0,0,0.19);' +
					'animation-name: animatetop;' +
					'animation-duration: 0.4s' +
				'}' +
				
				'@keyframes animatetop {' +
					'from {top: -300px; opacity: 0}' +
					'to {top: 0px; opacity: 1}' +
				'}'
  
	
	chrome.tabs.query({active: true, currentWindow: true}, function(tabs) {
		chrome.tabs.executeScript(
			null,
			{code: "var div=document.createElement('div'); document.body.prepend(div); div.innerHTML='"+modal+"'; var style=document.createElement('style'); document.body.prepend(style); style.innerHTML='"+modal_style+"';"}
		);
	});
	
  };
  