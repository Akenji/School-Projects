import { useContext, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { AuthContext } from "../Context/AuthContext";

const Form = ({mode}) => {
    const [username,setUsername]=useState("")
    const [email,setEmail]=useState("")
    const [password,setPassword]=useState("")
    const [confirm,setConfirm]=useState("")
    const [noUser,setNoUser]=useState(false)
    const [diffPasswords,setDiffPasswords]=useState(false)
    const {userState,updateState}=useContext(AuthContext)
    const navigate=useNavigate()
    const handleSubmit=(e)=>{
        e.preventDefault()
        if(password===confirm || mode=='login'){
            fetch('http://localhost:5000/'+mode,{
                method:'POST',
                credentials:'include',
                headers:{
                    'Content-Type':'application/json'
                },
                body:JSON.stringify({username,email,password})
            })
            .then((res)=>{
                if(res.ok) return res.json()
                    throw Error("Could't register user")
            })
            .then((data)=>{
                if(data.User){
                    updateState(true,data.User,data.ID,data.token,data.tasks)
                    localStorage.setItem("hooli",JSON.stringify({isLoggedIn:true,username:data.User,id:data.ID,token:data.token,posts:data.tasks}))
                    let d=new Date()
                    d.setTime(d.getTime()+24*60*60*1000)
                    d.toUTCString()
                    document.cookie=`hooli=${data.token};expires=${d}`
                    navigate('/')
                }
                if(!data.User){
                    setNoUser(true)
                }
            })
            .catch((error)=>{
                console.log(error)
            })
        }
        if(!(password===confirm)){
            setDiffPasswords(true)
        }
    }
    return ( 
        <div className="form mt-4 p-3">
            <form action="" className="row p-3" id="form">
                <div className="col-6 bg-1 p-5 form-left">
                <p className="form-text text-warning">{noUser && "No such user. Create account"}</p>
                {
                    (mode=="register") &&
                    <div className="form-group my-2">
                        <label htmlFor="username" className="h5 my-1">Username:</label>
                        <input type="text" name="username" className="form-control" value={username} onChange={(e)=>setUsername(e.target.value)} placeholder="Enter username" />
                    </div>
                }    
                    <div className="form-group my-2">
                        <label htmlFor="email" className="h5 my-1">Email:</label>
                        <input type="email" name="email" className="form-control" value={email} onChange={(e)=>{setEmail(e.target.value),setNoUser(false)}} placeholder="Enter email" />
                    </div>
                    <div className="form-group my-2">
                        <label htmlFor="password" className="h5 my-1">Password</label>
                        <input type="password" name="password" className="form-control"  value={password} onChange={(e)=>{setPassword(e.target.value),setNoUser(false)}} placeholder="Enter password" />
                    </div>
                {
                    (mode=="register") &&
                    <div className="form-group my-2">
                        <label htmlFor="confirmPassword" className="h5 my-1">Confirm Password</label>
                        <input type="password" name="confirmPassword" className="form-control" value={confirm} onChange={(e)=>{setConfirm(e.target.value),setDiffPasswords(false)}} placeholder="Enter password again" />
                        <div className="form-text text-warning">{diffPasswords && "Password should be same as above"}</div>
                    </div>
                }   
                    <div className="form-group my-2">
                        <button className="btn btn-custom w-100 h5" onClick={(e)=>handleSubmit(e)}>{(mode=="register")?"Register":"Login"}</button>
                    </div>
                    {
                        (mode=="register")?
                        <div className="form-text">Already have an account? <Link to='/login' className="h6">Login here</Link></div>
                        :
                        <div className="form-text" onClick={()=>setNoUser(false)}>Don't have an account? <Link to='/register' className="h6">Register here</Link></div>
                    }
                    {/* <div className="form-text">Don't have an account? <Link to='/register'>Register here</Link></div>
                    <div className="form-text">Already have an account? <Link to='/login'>Login here</Link></div> */}
                </div>
                <div className="col-6 bg-3 d-flex justify-content-center align-items-center p-5 form-right">
                    <div className="">
                        <p className="h5">{(mode=="register")?"Register":"Login"} with</p>
                        <p className="d-flex justify-content-center h5"> 
                            {/* <span className="fab fa-google google me-1"></span> */}
                            <a href="http://localhost:5000/auth/google" className="">
                                <span className="fab fa-google google me-1"></span>
                            </a>
                            <span className="fab fa-github google ms-1"></span>
                        </p>
                    </div>
                </div>
            </form>
        </div>
     );
}
 
export default Form;