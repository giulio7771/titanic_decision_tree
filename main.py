import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn import tree

def app():
    #lendo os dados no arquivo csv
    df = pd.read_csv("titanic.csv")
    
    #retirando as colunas irrelevantes para o contexto
    inputs = df.drop(['Name', 'Ticket', 'Cabin', 'PassengerId'],axis='columns')

    #removendo registros com informação faltante
    inputs = inputs.dropna()

    #isolando a coluna de sobreviventes
    target = inputs['Survived']

    #retirando a coluna de sobreviventes que queremos prever pela Árvore de Decisão
    inputs = inputs.drop('Survived',axis='columns')

    #codificando a coluna de tarifa paga para classes abrangendo faixas de valores
    encoded_fare = []
    for i in inputs['Fare']:
        if i > 31:
            fare_class = 3
        elif i > 14:
            fare_class = 2
        else:
            fare_class = 1
        encoded_fare.append(fare_class)
    inputs['fare_n'] = encoded_fare

    #codificando a coluna embarque para classes numericas
    encoded_embarked = []
    for i in inputs['Embarked']:
        if i == "S":
            e_class = 1
        elif i == "C":
            e_class = 2
        else:
            e_class = 3
        encoded_embarked.append(e_class)
    inputs['embarked_n'] = encoded_embarked

    #codificando as colunas textuais para numerico
    le_sex = LabelEncoder()
    #le_embarked = LabelEncoder()

    #adicionando a coluna cofidicada respectiva
    inputs['sex_n'] = le_sex.fit_transform(inputs['Sex'])
    #inputs['embarked_n'] = le_embarked.fit_transform(inputs['Embarked'])

    #removendo as colunas não codificadas
    inputs_n = inputs.drop(['Sex','Embarked', 'Fare'],axis='columns')

    model = tree.DecisionTreeClassifier()

    model.fit(inputs_n, target)
    model.score(inputs_n,target)

    #Entradas da predição da Árvore de Decisão:
    #Pclass, Age, SibSp, Parch, Fare (codificado), embarked(codificado), sex (codificado)

    #Braund, Mr. Owen Harris | Morto	
    passenger1 = [3, 22.0, 1, 0, 1, 1, 1]
    survived_cod = model.predict([passenger1])
    survived = survivedDecode(survived_cod)
    print("\nPassageiro 1: Braund, Mr. Owen Harris | Morto")
    print("Decision Tree: {}".format(survived))

    #Cumings, Mrs. John Bradley (Florence Briggs Thayer)	| Sobrevivente
    passenger2 = [1, 35.0, 1, 0, 3, 1, 0]
    survived_cod = model.predict([passenger2])
    survived = survivedDecode(survived_cod)
    print("\nPassageiro 2: Mrs. John Bradley | Sobrevivente")
    print("Decision Tree: {}".format(survived))

def survivedDecode(survived):
    if survived[0] == 0:
        return "Morto"
    else:
        return "Sobrevivente"

app()