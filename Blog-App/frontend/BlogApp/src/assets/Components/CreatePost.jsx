import { useContext, useState } from "react";
import { AuthContext } from "../Context/AuthContext";

const CreatePost = () => {
    const [title,setTitle]=useState("")
    const [body,setBody]=useState("")
    const [file,setFile]=useState()
    const {userState,updateState}=useContext(AuthContext)
    const handleCreate=(e)=>{
        e.preventDefault()
        let formData=new FormData()
        let image=document.querySelector('#image').files[0]
        formData.append("title",title)
        formData.append("body",body)
        formData.append("image",image)
        if(userState.username!='Guest'){
            let newPosts=[...userState.posts]
            newPosts.unshift({
                id:"",
                title:title,
                body:body,
                image:image.name,
                postedBy:userState.id
            })
            updateState(true,userState.username,userState.id,userState.token,newPosts)
            formData.append("tasks",newPosts)
            fetch('http://localhost:5000/task/add/'+userState.id,{
                method:'POST',
                credentials:'include',
                body:formData
            })
            .then((res)=>{
                if(res.ok) return res.json()
                throw Error("Couldn't save")
            })
            .then((data)=>{
                newPosts[0].id=data.post._id
                updateState(true,userState.username,userState.id,userState.token,newPosts)
                setTitle("")
                setBody("")
                setFile(null)
            })
            .catch((error)=>{
                console.log(error)
            })
        }else{
            updateState(false,"Guest","","",{postTitle:title,postBody:body})
            console.log(userState)
        }
    }
    return ( 
        <div className="create my-4">
            <h4>Create Post</h4>
            <form action="" onSubmit={(e)=>{handleCreate(e)}} encType='multipart/form-data'>
                <div className="form-group mb-2">
                    <label htmlFor="title">Post title:</label>
                    <input type="text" className="form-control" value={title} onChange={(e)=>setTitle(e.target.value)} name="title" placeholder="Enter post title" />
                </div>
                <div className="form-group mb-2">
                    <label htmlFor="body">Post Body:</label>
                    {/* <input type="text" className="form-control" name="body" placeholder="Enter post body" /> */}
                    <textarea name="body" className="form-control" id="" value={body} onChange={(e)=>setBody(e.target.value)} placeholder="Enter post body"></textarea>
                </div>
                <div className="form-group">
                    <label htmlFor="image">Post Image:</label>
                    <input type="file" name='image' id="image" className="form-control" onChange={(e)=>setFile(URL.createObjectURL(e.target.files[0]))} placeholder="Upload image if any" />
                    <div className="form-text">Upload post image if any.</div>
                    <img src={file} alt="" className="img-fluid w-75" />
                </div>
                <button className="btn btn-custom my-2" type="submit" >Create</button>
            </form>
        </div>
     );
}
 
export default CreatePost;