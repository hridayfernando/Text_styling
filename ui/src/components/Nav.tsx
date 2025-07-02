import * as React from 'react';

const Nav = () => {
    return (
        <>
            <nav className='flex w-[100vw] justify-between bg-gray-800 text-white
            p-5'>
                <div className="logo">Logo</div>


                <ul className='flex gap-2'>
                    <li><button>click me</button></li>
                    <li><button>about</button></li>
                    <li><button>contact us</button></li>
                </ul>
            </nav>
        </>
    )
}


export default Nav