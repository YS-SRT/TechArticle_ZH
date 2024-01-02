## 问题：

FastAPI中利用Dependency Inject的方式，可以在方法被调用之前做条件检查。当然Dependency Inject可以做的事情很多，比如只初始化一次调用训练好的AI模型。

排除登录接口，避免检查Token。可以用API编排的方式，不设置Dependency。也可以通过继承OAuth2PasswordBearer这个callable类来改写其中__Call__方法的逻辑。

## 方案:
１、简单编程：利用APIRouter(dependencies＝　)　或　include_router(dependencies＝　)　中设置，建议使用后者在统一的地方进行处理。

２、继承改写：继承OAuth2PasswordBearer类，改写其中的验证逻辑。

**Talk is cheap， Show code.**
## 代码：

方案１：把不需要验证Token的接口安排在同一个APIRouter中

```
check_token = OAuth2PasswordBearer("/login")

app = FastAPI()
app.include_router(loginAPI, tags=["Login"])
app.include_router(userAPI, prefix="/user", tags=["User"], dependencies=[Depends(check_token)])
app.include_router(prodAPI, prefix="/prod", tags=["Prod"], dependencies=[Depends(check_token)])
......
```

方案２：继承改写OAuth2PasswordBearer类，无须考虑Path

```
class SecurityIgnorePathEnum(str, Enum):
    login = "/user/login"
    apilogin="/user/apilogin"
    docs = "/docs"
    openapi = "/openapi.json"

class VerifyTokenMiddleware(OAuth2PasswordBearer):
    def __init__(self, tokenUrl):
        super().__init__(
            tokenUrl= tokenUrl,
            auto_error= True
        )

    async def __call__(self, request: Request) -> str | None:
        path:str = request.get("path")

        if path.startswith(tuple(member.value for member in SecurityIgnorePathEnum)):
            return None
        authorization:str = request.headers.get("Authorization")
        scheme,playload = get_authorization_scheme_param(authorization)
        if not authorization or scheme.lower() != "bearer":
                raise HTTPException(
        　　　　　　　　status_code=status.HTTP_401_UNAUTHORIZED,
        　　　　　　　　detail="Could not validate credentials",
        　　　　　　　　headers={"WWW-Authenticate": "Bearer"},)
        return playload
```
全局引入依赖注入

```
check_token = VerifyTokenMiddleware("/user/login")

app = FastAPI(dependencies=[Depends(check_token)])
app.include_router(userAPI, prefix="/user", tags=["User"])
app.include_router(prodAPI, prefix="/prod", tags=["Prod"])
```
## 完整代码地址
![my_github.jpg](https://p3-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/a6b93e2ca4b94fd9b80c216fa8485284~tplv-k3u1fbpfcp-jj-mark:0:0:0:0:q75.image#?w=1226&h=457&s=66411&e=jpg&b=fefefe)