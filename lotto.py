from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models
import matplotlib.pyplot as plt


def numbers2ohbin(numbers):

    ohbin = np.zeros(45)

    for i in range(6):
        ohbin[int(numbers[i])-1] = 1
    
    return ohbin

def ohbin2numbers(ohbin):

    numbers =[]

    for i in range(len(ohbin)):
        if ohbin[i] == 1.0:
            numbers.append(i+1)
    
    return numbers

def calc_reward(true_numbers, true_bonus, pred_numbers, mean_prize):

    count = 0

    for ps in pred_numbers:
        if ps in true_numbers:
            count += 1

    if count == 6:
        return 0, mean_prize[0]
    elif count == 5 and true_bonus in pred_numbers:
        return 1, mean_prize[1]
    elif count == 5:
        return 2, mean_prize[2]
    elif count == 4:
        return 3, mean_prize[3]
    elif count == 3:
        return 4, mean_prize[4]

    return 5, 0

def gen_numbers_from_probability(nums_prob):

    ball_box = []

    for n in range(45):
        ball_count = int(nums_prob[n] * 100 + 1)
        ball = np.full((ball_count), n+1) #1부터 시작
        ball_box += list(ball)

    selected_balls = []

    while True:
        
        if len(selected_balls) == 6:
            break
        
        ball_index = np.random.randint(len(ball_box), size=1)[0]
        ball = ball_box[ball_index]

        if ball not in selected_balls:
            selected_balls.append(ball)

    return selected_balls

def main():
    rows = np.loadtxt("./lotto.csv", delimiter=",", encoding='UTF8')
    row_count = len(rows)
    print(row_count)

    numbers = rows[:, 1:7]
    ohbins = list(map(numbers2ohbin, numbers))

    x_samples = ohbins[0: row_count-1]
    y_samples = ohbins[1: row_count]

    train_idx = (0, 800)
    val_idx = (801, 900)
    test_idx = (901, len(x_samples))
    
    model = keras.Sequential([
        keras.layers.LSTM(128, batch_input_shape=(1, 1, 45), return_sequences=False, stateful=True),
        keras.layers.Dense(45, activation='sigmoid')
    ])

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    train_loss = []
    train_acc = []
    val_loss = []
    val_acc = []
    
    # 최대 100번 에포크까지 수행
    for epoch in range(100):

        model.reset_states() # 중요! 매 에포크마다 1회부터 다시 훈련하므로 상태 초기화 필요

        batch_train_loss = []
        batch_train_acc = []
        
        for i in range(train_idx[0], train_idx[1]):
            
            xs = x_samples[i].reshape(1, 1, 45)
            ys = y_samples[i].reshape(1, 45)
            
            loss, acc = model.train_on_batch(xs, ys) #배치만큼 모델에 학습시킴

            batch_train_loss.append(loss)
            batch_train_acc.append(acc)

        train_loss.append(np.mean(batch_train_loss))
        train_acc.append(np.mean(batch_train_acc))

        batch_val_loss = []
        batch_val_acc = []

        for i in range(val_idx[0], val_idx[1]):

            xs = x_samples[i].reshape(1, 1, 45)
            ys = y_samples[i].reshape(1, 45)
            
            loss, acc = model.test_on_batch(xs, ys) #배치만큼 모델에 입력하여 나온 답을 정답과 비교함
            
            batch_val_loss.append(loss)
            batch_val_acc.append(acc)

        val_loss.append(np.mean(batch_val_loss))
        val_acc.append(np.mean(batch_val_acc))

        print('epoch {0:4d} train acc {1:0.3f} loss {2:0.3f} val acc {3:0.3f} loss {4:0.3f}'.format(epoch, np.mean(batch_train_acc), np.mean(batch_train_loss), np.mean(batch_val_acc), np.mean(batch_val_loss)))

    fig, loss_ax = plt.subplots()

    acc_ax = loss_ax.twinx()

    loss_ax.plot(train_loss, 'y', label='train loss')
    loss_ax.plot(val_loss, 'r', label='val loss')

    acc_ax.plot(train_acc, 'b', label='train acc')
    acc_ax.plot(val_acc, 'g', label='val acc')

    loss_ax.set_xlabel('epoch')
    loss_ax.set_ylabel('loss')
    acc_ax.set_ylabel('accuray')

    loss_ax.legend(loc='upper left')
    acc_ax.legend(loc='lower left')

    plt.show()

    mean_prize = [ np.mean(rows[87:, 8]),
           np.mean(rows[87:, 9]),
           np.mean(rows[87:, 10]),
           np.mean(rows[87:, 11]),
           np.mean(rows[87:, 12])]

    print(mean_prize)   

    train_total_reward = []
    train_total_grade = np.zeros(6, dtype=int)

    val_total_reward = []
    val_total_grade = np.zeros(6, dtype=int)

    test_total_reward = []
    test_total_grade = np.zeros(6, dtype=int)

    model.reset_states()

    print('[No. ] 1st 2nd 3rd 4th 5th 6th Rewards')

    for i in range(len(x_samples)):
        xs = x_samples[i].reshape(1, 1, 45)
        ys_pred = model.predict_on_batch(xs) # 모델의 출력값을 얻음
        
        sum_reward = 0
        sum_grade = np.zeros(6, dtype=int) # 6등까지 변수

        for n in range(10): # 10판 수행
            numbers = gen_numbers_from_probability(ys_pred[0])
            
            #i회차 입력 후 나온 출력을 i+1회차와 비교함
            grade, reward = calc_reward(rows[i+1,1:7], rows[i+1,7], numbers, mean_prize) 
            
            sum_reward += reward
            sum_grade[grade] += 1

            if i >= train_idx[0] and i < train_idx[1]:
                train_total_grade[grade] += 1
            elif i >= val_idx[0] and i < val_idx[1]:
                val_total_grade[grade] += 1
            elif i >= test_idx[0] and i < test_idx[1]:
                val_total_grade[grade] += 1
        
        if i >= train_idx[0] and i < train_idx[1]:
            train_total_reward.append(sum_reward)
        elif i >= val_idx[0] and i < val_idx[1]:
            val_total_reward.append(sum_reward)
        elif i >= test_idx[0] and i < test_idx[1]:
            test_total_reward.append(sum_reward)
                            
        print('[{0:4d}] {1:3d} {2:3d} {3:3d} {4:3d} {5:3d} {6:3d} {7:15,d}'.format(i+1, sum_grade[0], sum_grade[1], sum_grade[2], sum_grade[3], sum_grade[4], sum_grade[5], int(sum_reward)))

    print('Total') 
    print('==========')    
    print('Train {0:5d} {1:5d} {2:5d} {3:5d} {4:5d} {5:5d} {6:15,d}'.format(train_total_grade[0], train_total_grade[1], train_total_grade[2], train_total_grade[3], train_total_grade[4], train_total_grade[5], int(sum(train_total_reward))))
    print('Val   {0:5d} {1:5d} {2:5d} {3:5d} {4:5d} {5:5d} {6:15,d}'.format(val_total_grade[0], val_total_grade[1], val_total_grade[2], val_total_grade[3], val_total_grade[4], val_total_grade[5], int(sum(val_total_reward))))
    print('Test  {0:5d} {1:5d} {2:5d} {3:5d} {4:5d} {5:5d} {6:15,d}'.format(test_total_grade[0], test_total_grade[1], test_total_grade[2], test_total_grade[3], test_total_grade[4], test_total_grade[5], int(sum(test_total_reward))))
    print('==========')

    # 마지막 회차까지 학습한 모델로 다음 회차 추론

    print('receive numbers')

    xs = x_samples[-1].reshape(1, 1, 45)

    ys_pred = model.predict_on_batch(xs)

    list_numbers = []

    for n in range(10):
        numbers = gen_numbers_from_probability(ys_pred[0])
        numbers.sort()
        print('{0} : {1}'.format(n, numbers))
        list_numbers.append(numbers)




if __name__ == "__main__":
    main()