function injectS5Button(parentId, moduleId, dateSelector, resultDivId, style) {
  var parent = document.getElementById(parentId);
  if (!parent) return;
  style = style || '';
  var btn = document.createElement('button');
  btn.className = 'btn btn-sm';
  btn.style.cssText = 'margin-left:8px;background:#8b6914;color:#fff;border:none;border-radius:6px;padding:6px 14px;font-size:13px;font-weight:600;cursor:pointer' + (style ? ';' + style : '');
  btn.textContent = '\uD83D\uDD0D ' + t('AI Analysis');
  btn.onclick = function() { runModuleS5Analysis(moduleId, dateSelector, resultDivId); };
  parent.appendChild(btn);
}

function injectS5ResultDiv(targetId, resultDivId) {
  var target = document.getElementById(targetId);
  if (!target) return;
  var div = document.createElement('div');
  div.id = resultDivId;
  div.style.cssText = 'display:none;background:#fdfaf5;border:1px solid #d4c5a9;border-radius:10px;padding:16px;margin-bottom:12px;box-shadow:0 2px 8px rgba(139,105,20,0.08)';
  target.parentNode.insertBefore(div, target);
}

async function runModuleS5Analysis(moduleId, dateSelector, resultDivId) {
  var resDiv = document.getElementById(resultDivId);
  if (!resDiv) return;
  resDiv.style.display = 'block';
  resDiv.innerHTML = '<div style="text-align:center;padding:24px;color:#8b6914"><span class="spinner"></span> ' + t('Analyzing...') + '</div>';
  try {
    var hdrs = {'Content-Type': 'application/json', 'Authorization': 'Bearer ' + token};
    var dateEl = document.getElementById(dateSelector);
    var date = dateEl ? dateEl.value : '';
    var lang = localStorage.getItem('bakery_lang') || 'en';
    var r = await fetch(S5_API + '/analyze/module', {method: 'POST', headers: hdrs, body: JSON.stringify({module: moduleId, date: date, lang: lang})});
    if (!r.ok) { var txt = await r.text(); throw new Error(txt); }
    var d = await r.json();
    var html = '<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px;padding-bottom:8px;border-bottom:1px solid #e0d5c7"><h4 style="margin:0;color:#5d4037;font-size:15px">\uD83E\uDDE0 ' + t('AI Analysis') + '</h4><button onclick="document.getElementById(\'' + resultDivId + '\').style.display=\'none\'" style="background:none;border:none;color:#999;cursor:pointer;font-size:18px">&times;</button></div>';
    var summaryHtml = d.summary
      .replace(/\n\n/g, '</p><p style="font-size:14px;color:#3d322b;line-height:1.8;margin-bottom:6px">')
      .replace(/\n/g, '<br>')
      .replace(/^(BOTTOM LINE|WHY THIS HAPPENED|WHAT TO DO)/gm, '<strong style="color:#5d4037;font-size:15px"></strong>');
    html += '<div style="margin-bottom:12px"><p style="font-size:14px;color:#3d322b;line-height:1.8;margin-bottom:6px">' + summaryHtml + '</p></div>';
    // Key Factors are now synthesized into the summary by LLM ? no separate display needed
    if (d.recommendations && d.recommendations.length > 0) {
      html += '<div><strong style="font-size:13px;color:#5d4037">' + t('Recommendations') + '</strong><div style="margin-top:6px">';
      for (var i = 0; i < Math.min(d.recommendations.length, 4); i++) {
        var rec = d.recommendations[i];
        var urgColor = rec.urgency === 'high' ? '#c0392b' : rec.urgency === 'medium' ? '#e67e22' : '#27ae60';
        var urgBg = rec.urgency === 'high' ? '#fdedec' : rec.urgency === 'medium' ? '#fef5e7' : '#eafaf1';
        html += '<div style="display:flex;align-items:flex-start;gap:8px;margin-bottom:6px;font-size:13px;color:#3d322b"><span style="background:' + urgBg + ';color:' + urgColor + ';border-radius:3px;padding:2px 8px;font-size:10px;font-weight:600;white-space:nowrap;flex-shrink:0">' + (rec.urgency || 'medium').toUpperCase() + '</span><div><div>' + rec.action + '</div>' + (rec.rationale ? '<div style="color:#6b5b4f;font-size:11px;margin-top:2px">' + rec.rationale + '</div>' : '') + '</div></div>';
      }
      html += '</div></div>';
    }
    resDiv.innerHTML = html;
  } catch (ex) {
    resDiv.innerHTML = '<div style="color:#c0392b;text-align:center;padding:12px">' + t('Analysis failed') + ': ' + ex.message + '</div>';
  }
}