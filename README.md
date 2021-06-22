# Playing Cartpole with Deep Q-Learning

Github repo link
---
https://github.com/1am9trash/NYCU_2021_Spring_AI_Final_Project

Introduction:
---
> introduce the problem you want to solve, explain why it is important to solve it; and indicate the method you used to solve it. add a concept figure showing the overall idea behind the method you are presenting

現今機器學習模型的數據多依賴於人工的標示，然而現實世界中，許多時候一個行為難以明確的評分，抑或是行為與回饋間有相當長的延時，使得模型難以在真實世界裡有好的成效。  
在這次專題中，我們試圖透過強化學習的方式，與真實世界互動，並逐漸學習不同環境下該有的行為。我們使用openAI的環境模擬CartPole的遊戲，嘗試透過DQN、Nature DQN、Double DQN等模型獲取高分。  

Related work:
---
> previous methods that have explored a similar problem

- Playing Atari with Deep Reinforcement Learing, 2013  
  提出Deep Q-learning（DQN）模型，結合強化學習與深度學習，透過神經網路模擬Q-table，解決MDP對於記憶體與運算資源的限制。
- Human-level Control Through Deep Reinforcement learning, 2015  
  改善DQN模型，使用兩個神經網絡模型分別用於訓練與預估（Nature DQN模型）。此舉解決DQN中，神經網路用自身的預估來訓練自身參數，相關度過高的問題。
- Deep Reinforcement Learning with Double Q-Learning, 2016  
  在此前的DQN模型中，Bellman Equation都是貪心選取最高的評分來做訓練，然而這將導致評分高估，偏差較大的問題。Double DQN基於Nature DQN的模型，在Bellman Equation選取評分時，改為用訓練的神經網路選取，避免評分的膨脹。

Methodology:
---
> Details of the proposed technical solution

- Deep Q-learning訓練流程：  
  在強化學習中，agent會持續地做出行為，並接受該行為的回饋（Reward），再透過這些資料進行神經網路的擬合。  
  ![](https://i.imgur.com/wUCIx99.jpg)
- Get Action：  
  - 重要性：  
    在強化學習中，agent能拿到哪些類型的transition無疑是重要的，如果agent時常做出錯誤的行為，拿到的資料勢必都是遊戲剛開始時的state，難以訓練到後續state該有的反應，因此行為的選取自然是重要的，在此我們採用強化學習中常用的epsilon-greedy算法。  
  - e-greedy算法：  
    在此算法中，選擇行為時，有 $\epsilon$ 的機率隨機選擇一個行為，有 $1 - \epsilon$ 的機率選擇當下model判定最好的行為，隨training持續進行， $\epsilon$ 會持續下降，直到達到一個閥值。在我們的模型中， $\epsilon$ 初始為0.5，閥值為0.01。  
  - e-greedy算法的意義：  
    在訓練之初，model的結果並不可信，因此e-greedy算法花費更多的時間去探索不同行為的回報（ $\epsilon$ 較大），而隨著擬合持續進行，model逐漸可信賴，因此選擇隨機行為的機率，亦即 $\epsilon$ 越來越低，轉而傾向選擇model判斷的最佳解，在此狀況下，agent的存活時間會更長，也得以探索到離起始更為遙遠的state。  

- Store Experience:  
  - 重要性：  
    在DQN中，如果每拿到一個transition就做神經網路的擬合，由於訓練的資料都來自最近state的原因，神經網路會過擬合現在的遊戲，並遺忘一段時間之前的state應該做什麼行為。為了避免此問題，DQN將所有的transition存在memory中，並在需要擬合時從中抽取，使訓練的資料在時序上呈現常態分佈。  
  - 儲存內容：  
    Transition (s, a, r, s')  

    | 代號 | 意義         |
    |:---- | ------------ |
    | s    | 現在的state  |
    | a    | 行為         |
    | r    | 行為的reward |
    | s'   | 下一個state  |
  - Memory的切分優化：  
    在Cartpole中，Reward的分佈並不平均，使遊戲繼續跟使遊戲結束的transition的比例相當懸殊，這會影響學習的bias，若是連續抽取的資料都沒有會使遊戲結束的數據，會導致不好的訓練結果。  
    為了解決這個問題我們將memory拆分成相等的兩塊，命名為pos_memory跟neg_memory，這兩個memory分別儲存該回合遊戲未結束／結束的transition，當抽取資料學習的時候，則從pos_memory跟neg_memory裡面各隨機取一半batch size數量的資料來學習，保證訓練數據的平衡。  
  - Reward Function的優化：  
    在openAI的CartPole環境中，Reward並不顯著，只有當行為導致遊戲結束時Reward為0，其餘皆為1，這樣的Reward使收斂困難，因此我們改變了Reward Function。  
    
    首先觀察在CartPole中，使遊戲結束的因素：  
    1. 平衡車離中心點太遠  
       ![](https://i.imgur.com/28HOoNa.png)  
    2. 平衡車傾斜超過15度  
       ![](https://i.imgur.com/j7AiN5G.png)  
    
    考慮結束因素，設計對應的function：  

    |                     | 公式                             | 解釋                            |
    | ------------------- | -------------------------------- | ------------------------------- |
    | position penalty    | $2^{abs(position)}$               | 隨離中心點距離增加penalty做指數成長 |
    | inclination penalty | $2^{abs(inclination) \times 10}$ | 隨傾角增加penalty做指數成長，乘以10保證對Reward的影響與position penalty對等 |
    | bias                | 60 | 將reward調回正數 |
    
    $if\ game\ end,\ then\ reward = -100$  
    $otherwise,\ then\ reward = bias - position\ penalty - inclination\ penalty$  
    
- Learn：  
  model學習分為兩個步驟：Replay抽取數據、Fit訓練神經網路。  
  - Replay：  
    如Store Experience中所述，DQN並不是拿到一個transition就做擬合，而是在需要learn的時候從memory中抽取數據。Replay就是這個抽取的步驟，它會在所有的資料中抽取batch size大小的數據，並以此做訓練。  
    之所以每次只採用部分資料做學習的原因，大致與批梯度下降法邏輯相似。採用全部的資料做學習，速度太慢，而選取一定大小的資料訓練，已經足夠保證相似的更新方向。  
  - Fit：  
    對Replay的資料，根據Bellman Equation做神經網路的擬合。  

- DQN模型中的神經網路架構：  
  在openAI的環境中，提供(1, 4)的state，因此我們不用透過卷積層跟池化層分析出圖像中的資訊，而得以直接建構一個簡單的MLP的網路作為模型。  
    
  | Layer        | Shape      | activation |
  | ------------ | ---------- | ---------- |
  | Dense Input  | (None, 4)  | ReLU       |
  | Dense 1      | (None, 64) | ReLU       |
  | Dense 2      | (None, 32) | ReLU       |
  | Dense Output | (None, 2)  | ReLU       |

    
Experiments:
---
> present here experimental results of the method you have implemented with plots, graphs, images and visualizations

分別實作DQN、Nature DQN、Double DQN，並在原本／優化後的Reward、原本／切分後的Memory上運行，觀察其成效。  

- Nature DQN  
  1. 從Reward的角度而言，可發現使用原本Reward的model都較晚才開始提高分數，推測是因為Reward分佈稀疏的原因。  
  2. 從Memory的角度來看，使用拆分Memory的model最終有較好的分數以及穩定度，符合Replay資料平均，bias較小的推論。  
  3. 觀察Learning跟Reward的關係，可發現即使在相同的Learning次數下，優化後的model依然有更高的分數。  
  
  ![](https://i.imgur.com/UJOjFbN.png)  
  ![](https://i.imgur.com/KpZdZ51.png)  

- DDQN  
  1. 從Reward角度來看，紫色曲線與綠色曲線採用新的Reward有得到明顯的改善。  
  2. 從Memory角度來看，也可看出紫色的穩定度及分數比綠色來的好，這應該是對memory優化過後，model比較不容易發生遺忘的原因。  
  3. 相較於Nature DQN，DDQN Learning跟Reward的關係沒這麼顯著，但依然可發現在同樣Learning次數下，紫色曲線明顯最為優秀。  
  
  ![](https://i.imgur.com/yLVGHaL.png)  
  ![](https://i.imgur.com/EQIEWvP.png)  
  
- DQN  
  1. 在DQN模型中，優化並沒有顯著的提升效能。  
  2. 猜測可能因為DQN神經網路相關性較強的原因，不適合過於複雜的Reward Function。  
  ![](https://i.imgur.com/VHyw2m3.png)  
  ![](https://i.imgur.com/YTSRj7q.png)  
  
- 實際情況：  
  由於在openAI中預設分數多於200分就算通關，因此我們訓練model時，每個episode最多進行200次互動。  
  我們實際用訓練完的model測試時，可以發現DQN的分數大概落在200分左右，剛好通關。DDQN的分數則落在400分左右，成果比DQN要好一些。Nature DQN適應性則更強，幾乎是可以穩定上萬分，甚至不會結束遊戲。  
  
  [](https://user-images.githubusercontent.com/69944614/121707285-fd03e580-cb08-11eb-8ace-e0984912a50d.mov)  

Conclusion:
---
> Take home message

修改Reward Function跟Memory的架構，並透過實作數據比較後，我們發現：  
1. 設計精良的Reward Function可以使Model學習的速度加快，但其餘的Reward Function最終也可以Model學到東西，只不過會花費更多episode。  
2. Replay的資料是否平均，會決定Model的穩定性，當資料分佈不均時，可能導致對遊戲初始state的遺忘。  

在實作與測試中，我們發現：  
1. 在測試神經網路的參數時，如果是取穩定高分時的參數會比突然高分時的參數來的適應性更強。  

References:
---
1. Volodymyr Mnih, Koray Kavukcuoglu, David Silver, Alex Graves, Ioannis Antonoglou, Daan Wierstra, Martin Riedmiller(2013). Playing Atari with Deep Reinforcement Learning.
2. Volodymyr Mnih, Koray Kavukcuoglu, David Silver(2015). Human-level control through deep reinforcement learning.
3. Hado van Hasselt, Arthur Guez, David Silver(2016). Deep Reinforcement Learning with Double Q-learning.
