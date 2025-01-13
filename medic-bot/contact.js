const form = document.querySelector("form");

const sendEmail = () => {
  Email.send({
    Host: "smtp.elasticemail.com",
    Username: "username",
    Password: "password",
    To: "miltonsone12@gmail.com",
    From: "you@isp.com",
    Subject: "This is the subject",
    Body: "And this is the body",
  }).then((message) => alert(message));
  console.log("email send successfully ");
};
 