var phone_max_width = 667;
var tablet_max_width = 1155;

var useIDX = false;
var useGLO = false;
var useFilter = false;
var useTOC = true;

var numberOfTOCItemsInRow = 4;
var useCustomTOCLinks = false;
var mobileTocDrilldown = true;

var useANDsearch = false;

var customlink_1_text = "Custom URL1";
var customlink_1_href = "http://";
var customlink_2_text = "Custom URL2";
var customlink_2_href = "http://";

var titleColor = "#ffffff";
var backgroundColor = "#3BBCE0";
var fontFamily = "Arial";

(function() {

	var rh = window.rh;

	//The side of the sidebar / mobile overlay
	rh.consts("SIDEBAR_STATE", ".e.sidebarstate");
	rh.model.publish(rh.consts("SIDEBAR_STATE"), false);
	model.publish(rh.consts('KEY_IS_RESPONSIVE'), true);
	

	//Set the TOC type for mobile: Regular (false: default) or Drill down (true).
	rh.model.publish(rh.consts("KEY_MOBILE_TOC_DRILL_DOWN"), mobileTocDrilldown);

	// Search highlight colors
	model.publish(rh.consts('KEY_SHOW_SCROLL_TO_TOP'), true);
  model.publish(rh.consts('KEY_SEARCH_HIGHLIGHT_COLOR'), "#FFFFFF");
  model.publish(rh.consts('KEY_SEARCH_BG_COLOR'), "#e6e6e6");
	model.publish(rh.consts('KEY_CUSTOM_BUTTONS_CONFIG'), [{"name":"Expand/Collapse All","id":"0","key":"11496","title":"Expand/Collapse All","image":"toc_desktop.png","onclick":"rh.model.publish(rh.consts('EVT_EXPAND_COLLAPSE_ALL'));return false;"},{"name":"Remove Highlight","id":"1","title":"Remove Highlight","image":"remove_hightlight.png","key":"11497","onclick":"rh.model.publish(rh.consts('EVT_REMOVE_HIGHLIGHT'));return false;"},{"name":"Print","id":"2","key":"11498","title":"Print","image":"print_desktop.png","onclick":"rh.model.publish(rh.consts('EVT_PRINT_TOPIC'));return false;"}])
	model.publish(rh.consts('KEY_DO_NOT_PRESERVE_AR'), false);

	//Number of search results to be loaded at once.
	rh.consts('MAX_RESULTS', '.l.maxResults');
	rh.consts('MAX_RESULTS', '.l.maxResults');
	rh.model.publish(rh.consts('MAX_RESULTS'), 15);

	/* This layout has search on every page */
	rh.model.publish(rh.consts("KEY_CAN_HANDLE_SEARCH"), true);

	//Set the media queries
	var screens = {
	  ios: {user_agent: /(iPad|iPhone|iPod)/g}
	};

  screens.phone =  { media_query: 'screen and (max-width: '+ phone_max_width +'px)' };
  if(phone_max_width === 0) {
    screens.tablet =  { media_query: 'screen and (max-width: '+ tablet_max_width +'px)' };
  } else {
    screens.tablet =  { media_query: 'screen and (min-width: '+ (phone_max_width + 1) +'px) and (max-width: '+ tablet_max_width +'px)' };
  }
  if(tablet_max_width === 0) {
    screens.desktop =  { media_query: 'screen and (min-width: '+ (phone_max_width || 1) +'px)' };
  } else {
    screens.desktop =  { media_query: 'screen and (min-width: '+ (tablet_max_width + 1) +'px)' };
  }

	rh.model.publish(rh.consts('KEY_SCREEN'), screens);
	model.publish(rh.consts('KEY_DEFAULT_SCREEN'), "desktop");

  features = rh.model.get(rh.consts('KEY_FEATURE')) || {};

  //Publish which panes are available
  features.toc = useTOC;
  features.idx = useIDX;
  features.glo = useGLO;
  features.filter = useFilter;
	features.andsearch = useANDsearch;
  rh.model.publish(rh.consts('KEY_FEATURE'), features);

	//Custom URLs schema
	var Link = "custom-link-";
	var menuSuffix = "-menu";
	var headerSuffix = "-header";
	var excludeDefault = "http://";

	//Set the links for 1 if needed
	var menu = document.getElementById(Link+'1'+menuSuffix);
	var header = document.getElementById(Link+'1'+headerSuffix);
	if(customlink_1_href.trim() != "" && customlink_1_href.trim() != excludeDefault) {
		var link = '<a href="'+customlink_1_href+'" title="'+customlink_1_text+'" target="_blank">'+customlink_1_text+'</a>';
		menu.innerHTML = link;
		header.innerHTML = link;
	} else {
		menu.parentNode.removeChild(menu);
		header.parentNode.removeChild(header);
	}
	//Set the links for 2 if needed
	menu = document.getElementById(Link+'2'+menuSuffix);
	header = document.getElementById(Link+'2'+headerSuffix);
	if(customlink_2_href.trim() != "" && customlink_2_href.trim() != excludeDefault) {
		var link = '<a href="'+customlink_2_href+'" title="'+customlink_2_text+'" target="_blank">'+customlink_2_text+'</a>';
		menu.innerHTML = link;
		header.innerHTML = link;
	} else {
		menu.parentNode.removeChild(menu);
		header.parentNode.removeChild(header);
	}
	rh.initIndigo();

}.call(this));
