# 面试题：apitable的快速盲盒部署与调试(Typescript + Java)

## 题目
1. apitabl的项目地址 http://github.com/apitable/apitable [ver1.4.0]
2. 技术栈：
- backend-server  [java8/11, SpringBoot2.7x][Java17中需要去掉服务的配置参数]
- room-server [NestJS]
- databus-server [多数据库中间层]
- gateway [Nginx]
- Redis
- RabbitMq
- mysql
- minio [OSS]
- 其他工具：liquibase(数据库变更管理)、JaCoCo(Java单元测试覆盖率) 等

3. 部署，调试，找bug

## 解题思路
1. 基于Docker的单一镜像部署
2. 修改暴露端口，对外提供每项服务的Port 【在不熟悉代码的情况下，以便单独调试任何其中一个服务】   
4. 展示backend-server的调试
5. 展示room-server的Automation逻辑(Action & trigger)

## 编写代码
环境：win10 + WSL2 + VSCode + Docket Desktop

1、先找个国内快速的Docker Registry镜像拉取 apitable/all-in-one:latest， 按照命令创建容器 
```
docker run -d --name apitable_test  --env ENABLE_QUEUE_WORKER=true -v ${PWD}/.data:/apitable -p 3306:3306 -p 5672:5672 -p 6379:6379 -p 80:80 -p 9000:9000 apitable/all-in-one:latest
```
其中环境变量 ENABLE_QUEUE_WORKER=true 打开 Automation功能

显然这几个端口还不足够调试 backend-server, 去到目录 
```
\\wsl$\docker-desktop-data\data\docker\containers
```
找到对应的容器，文件夹前面字母和 $ docker ps 命令中列出的一样。

【修改之前退出Docker Desktop，不然会被缓存覆盖掉。】

已暴露端口：
- Minio：9000
- DataBus: 8625
- MySQL: 3306
- Rabbitmq: 5672
- Redis: 6379
- Web: 80

增加暴露端口：
- BackendServer: 8091 (把容器中的服务绑定到 8091(你可以随便改)，本机调试要起 8081 端口)
- room-server: 3002，3333，3334 [可以修改为绑定其他端口，以便本机调试不冲突]

修改文件 hostconfig.json 和 config.v2.json 如下：

hostconfig.json中：
```
"PortBindings": {
		"3002/tcp": [
			{
				"HostIp": "0.0.0.0",
				"HostPort": "3002"
			}
		],
		"3306/tcp": [
			{
				"HostIp": "0.0.0.0",
				"HostPort": "3306"
			}
		],
		"3333/tcp": [
			{
				"HostIp": "0.0.0.0",
				"HostPort": "3333"
			}
		],
		"3334/tcp": [
			{
				"HostIp": "0.0.0.0",
				"HostPort": "3334"
			}
		],
		"5672/tcp": [
			{
				"HostIp": "0.0.0.0",
				"HostPort": "5672"
			}
		],
		"6379/tcp": [
			{
				"HostIp": "0.0.0.0",
				"HostPort": "6379"
			}
		],
		"80/tcp": [
			{
				"HostIp": "0.0.0.0",
				"HostPort": "80"
			}
		],
		"8081/tcp": [
			{
				"HostIp": "0.0.0.0",
				"HostPort": "8091"
			}
		],
		"9000/tcp": [
			{
				"HostIp": "0.0.0.0",
				"HostPort": "9000"
			}
		]
	}
```

config.v2.json 中：
```
"Config": {
		"Hostname": "b51ad05802e9",
		"Domainname": "",
		"User": "",
		"AttachStdin": false,
		"AttachStdout": true,
		"AttachStderr": true,
		"ExposedPorts": {
			"3002/tcp": {
			},
			"3306/tcp": {
			},
			"3333/tcp": {
			},
			"3334/tcp": {
			},
			"5672/tcp": {
			},
			"6379/tcp": {
			},
			"80/tcp": {
			},
			"8081/tcp": {
			}
		}
```

启动Docker Desktop，开启容器就可以访问http://localhost:80, 检查Automation Action&Trigger都可以正常使用。

2、调试backend-server

修改源代码中 backen-service/application/src/main/resource/application.yml

通常 api-docs或swagger-ui可以只开一个，以便支持knife4j
```
springdoc:
  api-docs:
    enabled: ${API_DOCS_ENABLED:true}
  swagger-ui:
    enabled: ${API_DOCS_ENABLED:true}
```

同级目录 default.properties 中

```
knife4j.enable=true
knife4j.setting.enable-open-api=true
knife4j.setting.language=en
```
然后可以开始愉快的调试了

![work.jpg](https://p6-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/6c6898e95c3647408dd7cc794d65cbcb~tplv-k3u1fbpfcp-jj-mark:0:0:0:0:q75.image#?w=2513&h=1371&s=836409&e=jpg&b=171717)

3、如何找bug

在上图的断点中，代码自定义了Annotation（ApiResource、GetResource、PostResource）在Controller初始化时收集提供的API接口信息集合，置于APIResourceFactory中，用于对访问的控制。其中因为路径配置的问题，存在这样的//space/capacity 路径记录。 

```
@ApiResource(path = "/")
public class SpaceController {

@GetResource(path = "/space/capacity", requiredLogin = false)
public ResponseData<SpaceCapacityVO> capacity() {
   

}

}
```
好在用户访问路径验证算法不是简单的字符串比较，让我们看看有多少这样的路径存在。

总共222个API中有43中这样的路径记录：

![bug.jpg](https://p9-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/5c048b21fe6d4c619aad9399afcf974b~tplv-k3u1fbpfcp-jj-mark:0:0:0:0:q75.image#?w=636&h=1153&s=226819&e=jpg&b=fefdfd)

4、Automation Action&Trigger 的机制和用法

package/room-server/src/automation文件夹，是有相关的Action & Trigger 的定义和操作，它们的区别时Trigger会携带一个datasheetId，以便和数据表进行关联。
- action 定义数据和提供一些内部可调用的方法
- controller，提供 Action & Trigger 相关API，比如创建和存库操作。
- service，调用 Action & Trigger 的服务

循着调用路径一路向上能够找到 packages/core/src/automation_manager 里的相关调用机制。理解机制之后，写一个Action也比较容易。


```
import { AutomationAction } from "./actions/decorators/automation.action.decorator";
import { IActionResponse } from "./actions/interface/action.response";
import { IBaseAction, IUiSchema } from "./actions/interface/base.action";
import { sendMail } from "./actions/sms/index";

@AutomationAction("SendWarnMail")
export class SendWarnMailAction implements IBaseAction {
  async endpoint(input: any): Promise<IActionResponse<any>> {
    
    const mailRequest = {mailServer: { domain: "smtp.sina.com", port:"25"},
                         account: "xxx", password: "yyy",
                         to: input.addressList, subject: "Warn Email",
                         message: input.warnContent, template: "-- by xxx" };

    return new Promise(resolve =>{
      sendMail(mailRequest).then(resp =>{
          resolve(resp)
         }) 
      });

  }

  getInputSchema() {
    return {
      type: "object",
      properties: {
        addressList: {
          type: "string",
          title: "addressList"
        },
        warnContent: {
          type: "string",
          title: "warnContent"
        },

      }
    }

  }
  getUISchema(): IUiSchema {

    return { };
  }
  getOutputSchema() {

    return {

     };
  }

}
```

## 完整代码地址
![2023-11-27_093952.jpg](https://p3-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/6540c30dbcdc422e91b87fc87d371219~tplv-k3u1fbpfcp-jj-mark:0:0:0:0:q75.image#?w=987&h=570&s=83093&e=jpg&b=fcfcfc)