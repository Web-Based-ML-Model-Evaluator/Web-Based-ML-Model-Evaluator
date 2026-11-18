export const SAMPLE_DATASETS = {
  iris: {
    filename: 'iris_classification.csv',
    task: 'classification',
    target: 'species',
    content: `sepal_length,sepal_width,petal_length,petal_width,species
5.1,3.5,1.4,0.2,setosa
4.9,3.0,1.4,0.2,setosa
4.7,3.2,1.3,0.2,setosa
4.6,3.1,1.5,0.2,setosa
5.0,3.6,1.4,0.2,setosa
5.4,3.9,1.7,0.4,setosa
4.6,3.4,1.4,0.3,setosa
5.0,3.4,1.5,0.2,setosa
4.4,2.9,1.4,0.2,setosa
4.9,3.1,1.5,0.1,setosa
7.0,3.2,4.7,1.4,versicolor
6.4,3.2,4.5,1.5,versicolor
6.9,3.1,4.9,1.5,versicolor
5.5,2.3,4.0,1.3,versicolor
6.5,2.8,4.6,1.5,versicolor
5.7,2.8,4.5,1.3,versicolor
6.3,3.3,4.7,1.6,versicolor
4.9,2.4,3.3,1.0,versicolor
6.6,2.9,4.6,1.3,versicolor
5.2,2.7,3.9,1.4,versicolor
6.3,3.3,6.0,2.5,virginica
5.8,2.7,5.1,1.9,virginica
7.1,3.0,5.9,2.1,virginica
6.3,2.9,5.6,1.8,virginica
6.5,3.0,5.8,2.2,virginica
7.6,3.0,6.6,2.1,virginica
4.9,2.5,4.5,1.7,virginica
7.3,2.9,6.3,1.8,virginica
6.7,2.5,5.8,1.8,virginica
7.2,3.6,6.1,2.5,virginica`,
  },

  housing: {
    filename: 'housing_regression.csv',
    task: 'regression',
    target: 'price',
    content: `area_sqft,bedrooms,bathrooms,age_years,price
1200,2,1,15,220000
1450,3,2,10,285000
1600,3,2,8,310000
1850,4,2.5,5,375000
2100,4,3,3,430000
2400,4,3.5,2,495000
950,1,1,25,165000
1100,2,1,20,195000
1300,3,1.5,18,240000
1750,3,2,12,325000
1900,4,2,7,360000
2250,4,3,4,450000
2600,5,3.5,1,540000
2800,5,4,1,595000
1050,2,1,22,185000
1350,3,2,14,260000
1500,3,2,11,295000
1700,3,2.5,9,340000
2000,4,2.5,6,395000
2300,4,3,3,470000
2500,5,3.5,2,515000
2700,5,4,1,570000
3100,5,4.5,0,660000
3500,6,5,0,750000
1150,2,1.5,16,210000`,
  },

  customer: {
    filename: 'customer_clustering.csv',
    task: 'clustering',
    target: null,
    content: `annual_income_k,spending_score,visit_frequency,avg_basket_size
15,39,2,15
16,81,8,42
17,6,1,10
18,77,7,38
19,40,3,18
20,76,7,40
21,94,9,55
23,94,9,52
24,73,6,35
25,72,6,36
28,82,7,44
30,15,2,12
33,92,8,50
34,17,2,14
39,61,5,28
40,55,5,25
42,52,4,22
44,50,4,20
46,42,4,19
48,46,4,21
60,49,4,24
65,55,5,27
70,56,5,30
75,40,3,18
85,88,8,48`,
  },
};

export function createSampleFile(type: 'iris' | 'housing' | 'customer'): File {
  const item = SAMPLE_DATASETS[type];
  const blob = new Blob([item.content], { type: 'text/csv;charset=utf-8;' });
  return new File([blob], item.filename, { type: 'text/csv' });
}
