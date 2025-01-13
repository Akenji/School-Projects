import { useContext, useState } from "react";
import { AuthContext } from "../Context/AuthContext";
import {io} from 'socket.io-client'

const MessageUser = ({user}) => {
    const {userState}=useContext(AuthContext)
    const [show,setShow]=useState(false)
    const [message,setMessage]=useState("")
    const [messageR,setMessageR]=useState("")
    const [count,setCount]=useState(0)
    const socket=io('http://localhost:5000')
    document.windowId=Math.round(Math.random()*10000)
    const sendMessage=(e)=>{
        e.preventDefault()
        let newElement=document.createElement('div')
        newElement.classList.add("col-10","d-flex","justify-content-end")
        newElement.innerHTML=`
            <p class='bg-custom text-display-self d-flex justify-content-end p-3 my-2'>${message}</p>
        `
        let textField=document.getElementById('text-field')
        let display=document.getElementById('display').insertBefore(newElement,textField)
        socket.emit("send_message",{message:message,windowId:document.windowId})
    }
    socket.once("receive_message",(data)=>{
            if(data.windowId!=document.windowId) {
                setMessageR(data.message)
                setMessage("")
                let newElement=document.createElement('div')
                newElement.classList.add("col-12","d-flex","justify-content-start")
                newElement.innerHTML=`
                    <p class='bg-custom-2 text-display d-flex justify-content-start p-3 my-2'>${data.message}</p>
                `
                let textField=document.getElementById('text-field')
                let display=document.getElementById('display').insertBefore(newElement,textField)
            }
        })
    return ( 
        <div>
        {  user._id != userState.id && 
        <div className="user row border-top border-bottom py-2 mx-2" key={user._id}>
                <div className="user-img col-2 d-flex justify-content-center align-items-center">
                    {/* <img src="" alt="" /> */}
                    <span className="fas fa-user user-pic fa-lg"></span>
                </div>
                <div className="user-info px-4 ps-1 py-2 col-7">
                    <p className="fw-bold mb-1">{user.username}</p>
                    <p className="text-muted mt-1">Last message</p>
                </div>
                <div className="action col-3 pe-0">
                    <button className="btn btn-custom mb-2 w-100" onClick={()=>setShow(!show)}>Message <span className="fas fa-keyboard"></span></button>
                    <button className="btn btn-custom mt-2 w-100" disabled> Call <span className="fas fa-phone"></span></button>
                </div>
                <div className="display col-12 mt-3 border row justify-content-end" id="display">
                {/* {messageR && <p className='bg-custom-2 text-display d-flex justify-content-start p-3 my-2'> {messageR}</p>} */}
                {
                    show &&
                    <div className="text-field col-12 px-3 py-3" id="text-field">
                        <form action="">
                            <div className="form-group row">
                                <textarea name="message" className="form-control message-box" value={message} onChange={(e)=>setMessage(e.target.value)} rows="1" id="" placeholder='Enter message'></textarea>
                                <button className="btn btn-custom col-2 ms-2" onClick={(e)=>sendMessage(e)}>Send<span className="fas fa-paper-plane"></span></button>
                            </div>
                            <div className="form-group">
                            </div>
                        </form>
                    </div>
                }
                </div>
        </div>
        }
        </div>
     );
}
 
export default MessageUser;