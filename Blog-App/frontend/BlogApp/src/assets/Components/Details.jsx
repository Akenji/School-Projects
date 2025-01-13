import { useContext } from "react";
import { useNavigate, useParams } from "react-router-dom";
import { AuthContext } from "../Context/AuthContext";

const Details = ({}) => {
    const {userState,updateState}=useContext(AuthContext)
    const {id}=useParams()
    let image=""
    let title=""
    let body=""
    const post=userState.posts.filter((post)=>{
        return post.id==id
    })
    if(post.length>0){
        image=post[0].image
        title=post[0].title
        body=post[0].body
    }
    const navigate=useNavigate()
    const handleDelete=()=>{
        let posts=userState.posts
        posts=posts.filter((post)=>{
            return post.id!=id
        })
        updateState(true,userState.username,userState.id,userState.token,posts)
        fetch('http://localhost:5000/task/delete/'+id,{
            method:'POST',
            credentials:'include',
            headers:{
                'Content-Type':'application/json'
            },
            body:JSON.stringify(posts)
        })
        .then((res)=>{
            if(res.ok) return res.json()
                throw Error("Couldn't delete")
        })
        .then((data)=>{
            console.log(data)
            if(data.Deleted){
                navigate('/')
            }
        })
        .catch((error)=>{
            console.log(error)
        })
    }
    return ( 
        <div className="container row mt-4">
            <div className="post w-100 shadow px-0">
                <div className="post-img w-100 mb-2">
                    <img src={"/assets/img/"+image} className="img-fluid" alt="" />
                </div>
                <div className="post-title p-2 px-3 h4">
                    {title}
                </div>
                <div className="post-body p-2 px-3">
                    {body}
                </div>
                <div className="options p-2 px-3">
                    <button className="btn btn-outline-warning me-2">Edit</button>
                    <button className="btn btn-danger ms-2" onClick={()=>handleDelete()}>Delete</button>
                </div>
            </div>
        </div>
     );
}
 
export default Details;