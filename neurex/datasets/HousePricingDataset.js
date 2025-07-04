
/** 

House pricing dataset for regression task

credits to: Patrick Metzdorf for the dataset
https://medium.com/@pat_metzdorf/building-a-basic-neural-net-using-javascript-1f554780dc60

*/

const normalize = (value, min, max) => (value - min) / (max - min);

function denormalize(value, min, max) {
    if ([value, min, max].some(v => typeof v !== 'number' || isNaN(v))) {
      console.error("Bad denormalize input:", value, min, max);
      return NaN;
    }
    return value * (max - min) + min;
}

// [size (sq ft), bedrooms, price ($)]
const raw = [
  [1400, 3, 200000],
  [1600, 3, 230000],
  [1700, 3, 245000],
  [1875, 4, 275000],
  [1100, 2, 180000],
  [2350, 4, 320000],
  [2100, 4, 305000],
  [1500, 3, 215000],
  [1250, 2, 190000],
  [1950, 4, 285000],
  [2000, 3, 290000],
  [1300, 3, 205000],
  [2200, 4, 310000],
  [1000, 2, 175000],
  [1750, 3, 250000],
  [1650, 3, 240000],
  [2400, 5, 330000],
  [1550, 3, 225000],
  [1450, 2, 210000],
  [1800, 3, 260000],
  [1150, 2, 185000],
  [2500, 4, 340000],
  [2050, 4, 295000],
  [1350, 3, 212000],
  [1900, 3, 270000],
  [1050, 2, 178000],
  [2300, 4, 318000],
  [1700, 4, 265000],
  [1600, 2, 220000],
  [2800, 5, 370000],
  [1200, 2, 188000],
  [2150, 4, 308000],
  [1500, 2, 208000],
  [1920, 3, 272000],
  [2600, 5, 350000],
  [1380, 3, 218000],
  [1080, 2, 182000],
  [2450, 4, 335000],
  [1780, 3, 255000],
  [1620, 3, 235000],
  [2700, 5, 360000],
  [1580, 3, 228000],
  [1420, 2, 205000],
  [1850, 3, 268000],
  [1120, 2, 183000],
  [2550, 4, 345000],
  [2020, 4, 298000],
  [1320, 3, 208000],
  [1980, 3, 280000],
  [1020, 2, 176000],
  [2380, 4, 325000],
  [1720, 4, 260000],
  [1680, 2, 222000],
];

const newHouses = [
    [1550, 3],
    [2000, 4],
    [1200, 2]
];

const sizeMin = Math.min(...raw.map(d => d[0]));
const sizeMax = Math.max(...raw.map(d => d[0]));
const bedMin = Math.min(...raw.map(d => d[1]));
const bedMax = Math.max(...raw.map(d => d[1]));
const priceMin = Math.min(...raw.map(d => d[2]));
const priceMax = Math.max(...raw.map(d => d[2]));

const trainX = raw.map(([size, bed]) => [normalize(size, sizeMin, sizeMax), normalize(bed, bedMin, bedMax)]);

const trainY = raw.map(([_, __, price]) => normalize(price, priceMin, priceMax));

const testX = newHouses.map(([size, bed]) => [
  normalize(size, sizeMin, sizeMax),
  normalize(bed, bedMin, bedMax)
]);

module.exports = {
  trainX,
  trainY,
  testX,
  priceMin,
  priceMax,
  normalize,
  denormalize
};
