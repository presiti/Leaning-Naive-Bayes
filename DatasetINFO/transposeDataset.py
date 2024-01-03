import pandas as pd
import numpy as np

def transpose(df, c_list, f_list):
    iris_trans=pd.DataFrame()
    print('transpose dataset-------------------------')
    print('> transpose dataset')
    for i in range(3):              # 0~2, class 수만큼 돌기
        print('transpose :', c_list[i])
        iris_temp = df[df.variety==c_list[i]].transpose()     # 1. class별로 떼어와서 transpose로 행과 열을 바꾸기
        iris_np=np.empty(200)                                           # class별 데이터를 담을 numpy 리스트
        
        for j in range(4):          # 0~3, feature 수만큼 돌기
            f_np=iris_temp.iloc[j].to_numpy()                                      # 2. feature 4개를 한 줄씩 떼오기
            print(f_list[j], ' first value : ', f_np[0])
            for k in range(50):
                iris_np[k+j*50]=f_np[k]                                 # 3. 값을 하나씩 꺼내어 넣어 clsas별 numpy 리스트로 만들기
        iris_trans[c_list[i]]=iris_np                                   # 4. 데이터 프레임에 컬럼명을 class명으로 정하여 numpy리스트 넣기
        print(c_list[i], 'done\n')
    print('> add feature column')
    iris_f=list()                                                       # feature 열 값을 담을 리스트
    for j in range(4):
        for k in range(50):
            iris_f.append(f_list[j])                                    # 5. feature열에 넣을 feature 이름 리스트 생성하기
        print('featrue name : ',iris_f[j*50])                           # feature 이름이 바뀌는 지점 값 출력해서 잘 들어가고 있는지 확인
    iris_trans['feature']=iris_f
    print()

    return iris_trans