$(document).ready(function() {

  // Initialize DataTables with FixedHeader
  var table = $('#comparisonTable').DataTable({
    dom: '<"top"f>rt<"bottom"lip><"clear">',
    responsive: false,
    pageLength: 50,
    deferRender: true,
    order: [[6, 'desc']], // default sort by value_new
    columnDefs: [
      { targets: [1, 2, 3], type: 'num' },          // shares columns
      { targets: [5, 6, 7], type: 'num-fmt' },      // value columns
      { targets: [4, 8], type: 'custom-percent' },  // your change % columns
    ],
    dom: '<"d-flex justify-content-start my-3"f>rtip',
    language: {
      searchPlaceholder: "Filter comparison results..."
    },
    rowCallback: function(row, data, index) {
        var sharesChange = data[4].trim();
        var valueChange = data[8].trim();

        // Initialize state for coloring
        var addGreen = false;
        var addRed = false;

        // Check shares_change_pct
        if (sharesChange === 'NEW') {
            addGreen = true;
        } else if (sharesChange === 'REMOVED') {
            addRed = true;
        }

        // Check value_change_pct
        if (valueChange === 'NEW') {
            addGreen = true;
        } else if (valueChange === 'REMOVED') {
            addRed = true;
        }
        // Add classes, green takes precedence over red if both are found
        $(row).removeClass('green-row red-row'); // Always clear first

        if (addGreen) {
            $(row).addClass('green-row');
        } else if (addRed) {
            $(row).addClass('red-row');
        }
    }
  });
  new $.fn.dataTable.FixedHeader(table);
});

// Custom sorting for percentage columns with NEW and REMOVED
jQuery.extend(jQuery.fn.dataTable.ext.type.order, {
  'custom-percent-pre': function (data) {
    if (data === 'NEW') {
      return Number.POSITIVE_INFINITY;  // highest value
    }
    if (data === 'REMOVED') {
      return Number.NEGATIVE_INFINITY;  // lowest value
    }
    // Remove % sign and parse float
    var num = parseFloat(data.replace('%', '').trim());
    return isNaN(num) ? 0 : num;
  }
});
