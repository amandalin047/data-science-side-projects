const pillCols = [5, 6, 7];
const pillColLetters = ["E", "F", "G"];
const DATES=1, C=3, B=4, PILL1=5, PILL2=6, PILL3=7, G=9, NOTES=11, COMMAND=12; 
const imageUrlT = "imageUrlT";
const imageUrlF = "imageUrlF";
const imageUrlM = "imageUrlM";

const instructions = `INSTRUCTIONS:
  - Click cells D8, I8, I9 to view values & comments;

COMMANDS (I'll type from phone):
  - In cell L8, type "t" to show image;
  - In cell L8, type "remove" to remove t image;
  - In cell L10, type "f" to show image;
  - In cell L10, type "remove" to remove f image;
  - In cell L16, type "done" to close INSTRUCTIONS.`

function updatePillCount (sheet, datesLastRow){
pillCols.forEach((col) => {
  let pillArray = sheet.getSheetValues(2, col, datesLastRow-1, 1).flat();
  let s = 0;
  let idx = pillArray.length - 6;
  pillArray = pillArray.slice(-7);
  pillArray.forEach((num) => {
    idx += 1;

    if (typeof num != "string"){
      s += num;  
      if (col === PILL1 && num >= 2){
        let pillCell = sheet.getRange("E" + idx.toString());
        pillCell.setValue(num + "💀");
      }  
      else if (col === PILL2 && num >= 3){
        let pillCell = sheet.getRange("F" + idx.toString());
        pillCell.setValue(num + "💀");
      }
    } else if (num.includes("💀")){
      num = parseInt(num.replace(/\D/g, ""));
      s += num;
    }
    }  
  );
let rangeToUpdate = sheet.getRange(pillColLetters[pillCols.indexOf(col)]+(datesLastRow+1).toString());
rangeToUpdate.setValue(s).setFontColor("#1E90FF");
});
}


function clearPillCount(sheet, lastRow){
  let wLastRow = sheet.getRange("G2:G"+lastRow.toString()).getValues().flat();
  let wMax = Math.max.apply(Math, wLastRow);
  if (wMax > 1) {
    wLastRow = (wLastRow.indexOf(wMax) + 2).toString();
  } else if (wMax === 1){
    let wLastRowReversed = [...wLastRow].reverse();
    let lastIdx = wLastRowReversed.length - (wLastRowReversed.indexOf(1) + 1);
    wLastRow = (lastIdx + 2).toString();
  } else {
    wLastRow = (wLastRow.indexOf(0) + 2).toString();
  }
  
  let rangeToClear = sheet.getRange("E"+wLastRow+":G"+wLastRow);
  let colors = rangeToClear.getFontColors().flat();

  if (colors.every(color => color === "#1e90ff")){
    rangeToClear.setValue(null);
  }
}

function showImage(sheet, row, column, url, width, height){
  sheet.setColumnWidths(column+1, column+1, width);
  sheet.setRowHeights(row, row, height);
  sheet.getRange(row, column+1).setFormula(`IMAGE("${url}", 4, ${height}, ${width})`);
}

function command(sheet, cell, row, column, oldVal, currentVal, lastRow, datesLastRow,
                urlT, urlF, urlM, instructions){
  if (!oldVal && currentVal==="t"){
    if (sheet.getRange(row, GYN)){
      showImage(sheet, row, column, urlT, 720, 480);
    }
    cell.clear();
  } else if (!oldVal && currentVal==="f"){
    showImage(sheet, row, column, urlF, 720, 480);
    cell.clear();   
  } else if (!oldVal && currentVal=="m"){
    showImage(sheet, row, column, urlM, 420, 750);
    cell.clear(); 
  } else if (!oldVal && currentVal==="remove"){
    let w = sheet.getColumnWidth(1);
    let h = sheet.getRowHeight(1);
    sheet.setColumnWidths(column+1, column+1, w);
    sheet.setRowHeights(row, row, h);
    sheet.getRange(row, column+1).clear();
    cell.clear();
  } else if (!oldVal && currentVal=="dates"){
    for (let i=1; i<=7; i++){
      let startCell = "A" + datesLastRow.toString();
      let plus = i.toString();
      sheet.getRange(datesLastRow+i, 1).setFormula(`${startCell} + ${plus}`);  
    }
    clearPillCount(sheet, lastRow);
    let borderlineUnder = sheet.getRange("A" + (datesLastRow+7).toString() + ":K" + (datesLastRow+7).toString());
    borderlineUnder.setBorder(false, false, true, false, false, false, null,
                              SpreadsheetApp.BorderStyle.SOLID_MEDIUM);
    cell.clear();
  } else if (!oldVal && currentVal=="instructions"){
    let instructionsCell = sheet.getRange(datesLastRow+2, NOTES);
    sheet.setRowHeights(datesLastRow+2, datesLastRow+2, 147);
    instructionsCell.setValue(instructions).setFontSize(12);
    cell.clear();
  } else if (!oldVal && currentVal=="done"){
    sheet.getRange(datesLastRow+2, NOTES).setValue(null);
    let h = sheet.getRowHeight(1);
    sheet.setRowHeights(datesLastRow+2, datesLastRow+2, h);
    cell.clear();
  } else if (!oldVal && currentVal=="fill"){
    let wArray = sheet.getRange("G2:G"+(datesLastRow-7).toString()).getValues().flat();
    let arrayToFill = [];
    let count = 0;
    if (parseInt(wArray.slice(-1)) === 1){
      arrayToFill = [null,1,null,1,null,1,null];
      count = 3;
    } else {
      arrayToFill = [1,null,1,null,1,null,1];
      count = 4;
    }
    let i = 1
    arrayToFill.forEach((n) => {
        let c = sheet.getRange(datesLastRow-7+i, PILL3)
        c.setValue(n);
        c.setFontColor("black");
        i += 1;
    })
    sheet.getRange(datesLastRow+1, PILL3).setValue("sum="+count.toString());
    cell.clear();
  } else if (!oldVal && currentVal=="unfill"){
    let rangeToUnfill = sheet.getRange("G"+(datesLastRow-6).toString()+":G"+(datesLastRow+1).toString());
    rangeToUnfill.setValue(null);
    cell.clear();
  }
}


function onEdit(e) {
  let sheet = e.source.getActiveSheet();
  let lastRow = sheet.getLastRow();
  let cell = e.range;
  let column = cell.getColumn();
  let row = cell.getRow();
  let oldValue = e.oldValue;
  let value = e.value;

  const dontExecute = [2, 8, 9, 10, 11];
  if (dontExecute.includes(column) || row === 1){
    return; 
  }

  let dates = sheet.getRange("A1:A"+lastRow.toString()).getValues().flat();
  let datesLastRow = 0;
  for(let i = lastRow; i > 0; i-=1){
    if (dates[i-1] != ""){
        datesLastRow = i;
        break;
    }
  }

if (column===B){
  // B column edited
  let cCell = sheet.getRange(row, 3);
  let cCellPrior = sheet.getRange(row-1, 3);
//checks the B edit is an added value not deletion of cell value
if (!oldValue && value==="v"){
  cell.setFontColor("red");
//check C occured same day or day prior
  if (cCell.getValue()==="v"){
      //change C to red
       cCell.setFontColor("red");
    } else if (cCellPrior.getValue()==="v"){
      cCellPrior.setFontColor("red");
    }
} else if (!value){
  if (cCell.getValue()==="v"){
    //change C back to black
    cCell.setFontColor("black");
    } else if (cCellPrior.getValue()==="v"){
    cCellPrior.setFontColor("black");
    }
}
} else if (column != C) {
  cell.setFontColor("black");
}

// if DATES edited manually, clear pill count and reset
if (column===DATES) {
  clearPillCount(sheet, lastRow);
  return;
}

// calculate total pills per week
if (pillCols.includes(column)){
  updatePillCount(sheet, datesLastRow);
}

//command control
if (column===COMMAND){
  command(sheet, cell, row, column, oldValue, value, lastRow, datesLastRow,
          imageUrlT, imageUrlF, imageUrlM, instructions);
}

}