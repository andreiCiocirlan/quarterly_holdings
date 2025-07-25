// Custom sort for values with $ and suffixes (M, B, T)
jQuery.extend(jQuery.fn.dataTable.ext.type.order, {
    // Define how to extract the numerical value from formatted string for sorting ascending
    "formatted-num-pre": function (data) {
        if (!data || data === '-' || typeof data !== 'string') {
            return 0;
        }
        // Remove $ and spaces
        data = data.replace(/\$/g, '').replace(/\s/g, '');

        // Multiplier based on suffix
        let multiplier = 1;
        if (data.endsWith('T')) {
            multiplier = 1e12;
            data = data.slice(0, -1);
        } else if (data.endsWith('B')) {
            multiplier = 1e9;
            data = data.slice(0, -1);
        } else if (data.endsWith('M')) {
            multiplier = 1e6;
            data = data.slice(0, -1);
        }

        // Parse float and multiply
        let num = parseFloat(data);
        return (isNaN(num) ? 0 : num * multiplier);
    },

    // Just return ascending or descending value
    "formatted-num-asc": function (a, b) {
        return a - b;
    },

    "formatted-num-desc": function (a, b) {
        return b - a;
    }
});

$(document).ready(function() {
  var table = $('#filersAumTable').DataTable({
    pageLength: 50,
    order: [[2, 'desc']],  // Sort by Holdings Value column (index 2 zero-based)
    columnDefs: [
      { targets: [1], type: 'num-fmt' },  // Holdings Count
      { targets: [2], type: 'formatted-num' }   // Holdings Value
    ],
    language: {
      searchPlaceholder: "Filter filers..."
    },
    responsive: true,
    dom: '<"top"f>rt<"bottom"lip><"clear">',
    deferRender: true
  });

  // Optional: fixed header while scrolling
  new $.fn.dataTable.FixedHeader(table);
});