const userSchema=require('../model/user')
const postSchema=require('../model/post')
const jwt=require('jsonwebtoken')
const fs=require('fs')

const maxAge=3600*24
const createToken=(id,username)=>{
    return jwt.sign({id,username},process.env.Secret,{expiresIn:maxAge})
}


const register=async(req,res)=>{
    try{
    }
    catch(error){
        console.log(error.message)
        res.json({Error:error.message})
    }
}
const login=async(req,res)=>{
    try{


    }catch (error){
        console.log(error.message)
        res.json({Error:error.message})
    }
}
const logout=(req,res)=>{
    try {

    } catch (error) {
        res.json({Error:error.message})
    }
}
const getPosts=async(req,res)=>{
    try{

    }catch(error){
        console.log(error.message)
        res.json({Error:error.message})
    }
}
const addNewPost=async (req,res)=>{
    try{

    }catch(error){
        console.log(error.message)
        res.json({Error:error.message})
    }
}
const deletePost=async (req,res)=>{
    try{

    }catch(error){
        console.log(error.message)
        res.json({Error:error.message}) 
    }
}

const getUsers=async (req,res)=>{
    try{

    }catch(error){
        console.log(error.message)
        res.json({Error:error.message}) 
    }
}

exports.register=register
exports.login=login
exports.logout=logout
exports.getPosts=getPosts
exports.addNewPost=addNewPost
exports.deletePost=deletePost
exports.getUsers=getUsers