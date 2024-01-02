# 面试题：卡牌的胜负判断(Vue3实现)

## 题目：

![Requirement.png](https://p3-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/49a3fccbf9374607b9e82c5c76994da5~tplv-k3u1fbpfcp-jj-mark:0:0:0:0:q75.image#?w=898&h=954&s=2575260&e=png&b=fcfcfc)


## 解题思路：
1. 洗牌【Fisher–Yates 算法】
2. 发牌【按顺序分发】
3. 判断结果 【查重、比较&排序、结果输出】
4. 重新开始下轮
5. 响应式API绑定

**Talk is cheap， Show code.**
## 编写代码
1、定义数据类型

```
export type Record = {
    roundId: number,  //第几轮发牌
    playerId: number, //玩家Id
    cards: string, //玩家手中的牌
    result: boolean  //  输赢判定结果 (win = 1 / lost = 0)
}

//存储于Vuex中的接口定义
export interface IResultInfo {
    history: Record [] 
}

//用于玩家手中的牌面比较和排序
export type SortedAnalysisData = {
   playerId: number
   prefix: string,  //前缀，注意 10 需要特别处理, A/J/Q 等
   count: number, //前缀重复次数
   suffixs: string //后缀联合的字符串，比如：^@*
}

```
2、定义常量类型
```
let cards = ["2@","2#","2^","2*",
      "3@","3#","3^","3*",
      "4@","4#","4^","4*",
      "5@","5#","5^","5*",
      "6@","6#","6^","6*",
      "7@","7#","7^","7*",
      "8@","8#","8^","8*",
      "9@","9#","9^","9*",
      "10@","10#","10^","10*",
      "J@","J#","J^","J*",
      "Q@","Q#","Q^","Q*",
      "K@","K#","K^","K*",
      "A@","A#","A^","A*",
    ];
    
//用于按顺序Index比较
const prefixSequence = ["2","3","4","5","6","7","8","9","10","J","Q","K","A",];
const suffixSequence = ["@", "#", "^", "*"];
    
```
3、洗牌 & 发牌

```
let curPlayerCards: string[][] = [[], [], [], []];  

//Fisher–Yates for random sequence
  function shuffleCards(): void {
      let currentIndex = cards.length;
      let randomIndex = 0;

      while (currentIndex !== 0) {
        randomIndex = Math.floor(Math.random() * currentIndex);
        currentIndex--;

        [cards[currentIndex], cards[randomIndex]] = [
          cards[randomIndex],
          cards[currentIndex],
        ];
      }
      
   function dealCard(): void {
      cards.forEach((value: string, index: number) => {
        curPlayerCards[index % 4][Math.trunc(index / 4)] = value;
      });
    }
    
 } 
 
```

4、查重 & 标识
```
  function findRepeatCards(playerId: number, cardArray: string[]) {
      cardArray.reduce((analysisData, cur) => {
        let prefix = cur.charAt(0);
        let suffix = cur.charAt(1);

        //10 is special
        if (cur.charCodeAt(1) >= 48 && cur.charCodeAt(1) <= 57) {
          prefix = cur.substring(0, 2);
          suffix = cur.charAt(2);
        }

        let matchedRecord = isExistedInAnalysisData(
          playerId,
          prefix,
          analysisData
        );
        if (typeof matchedRecord !== "undefined") {
          matchedRecord.count += 1;
          matchedRecord.suffixs += suffix;
        } else {
          analysisData.push({
            playerId: playerId,
            prefix: prefix,
            count: 1,
            suffixs: suffix,
          });
        }

        return analysisData;
      }, analysisData);
  }
  
 function isExistedInAnalysisData(
      playerId: number,
      prefix: string,
      analysisData: SortedAnalysisData[]
    ) {
      return analysisData.find(
        (data) => data.prefix === prefix && data.playerId === playerId
      );
    }
  
```

5、比较 & 排序

```
//玩家手中牌面的比较和排序

export function sortByPrefixAndSuffix(prefixSequence:string[], suffixSequence:string[], analysisData:SortedAnalysisData[]){

    analysisData.sort((first, second) =>{
        
         //max repeat count
         if(first.count > second.count){

            return -1;

          }else if (first.count === second.count){

            const firstPrefixIndex = prefixSequence.indexOf(first.prefix);
            const secondPrefixIndex = prefixSequence.indexOf(second.prefix);
            
            //large prefix index 
            if(firstPrefixIndex > secondPrefixIndex){
                
                return -1;

            }else if(firstPrefixIndex === secondPrefixIndex){

              const firstSuffixIndex = spliteAndCheckMaxIndex(first.suffixs, suffixSequence);
              const secondSuffixIndex = spliteAndCheckMaxIndex(second.suffixs, suffixSequence);
                
              //large suffix index
              if(firstSuffixIndex > secondSuffixIndex){
                  return -1;
              }
           }

         }

         return 1;

  })

}

function spliteAndCheckMaxIndex(suffixs:string, suffixSequence:string[]){
    let index = 0;
    suffixs.split('').forEach((item)=> {

        const temp = suffixSequence.indexOf(item);
        if(temp > index){

            index = temp;
        }

    })

    return index;
}
```

## 完整代码地址：[GitHub - YS-SRT/TechInterview](https://github.com/YS-SRT/TechInterview)

![my_github.jpg](https://p3-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/a6b93e2ca4b94fd9b80c216fa8485284~tplv-k3u1fbpfcp-jj-mark:0:0:0:0:q75.image#?w=1226&h=457&s=66411&e=jpg&b=fefefe)