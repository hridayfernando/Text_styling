import React, { useState, useCallback, DragEvent } from "react";
import brain from '../asssets/brain.svg';

// Custom Image type
export interface Image {
  id: string;
  file: File;
  preview: string;
}

const TrainerContainer: React.FC = () => {
  const [original_images, setOriginalImages] = useState<Image[]>([]);
  const [styled_images,setStyledImages] = useState<Image[]>([]);
  // Handle files from input or drop
  const processFiles = (files: FileList,isOriginal: boolean) => {
    const newImages: Image[] = [];
    Array.from(files).forEach((file) => {
      const id = `${file.name}-${file.size}-${Date.now()}`;
      const preview = URL.createObjectURL(file);
      newImages.push({ id, file, preview });
    });
    if (isOriginal){
      setOriginalImages((prev) => [...prev, ...newImages]);
    }
    else {
      setStyledImages((prev) => [...prev,...newImages])
    }
  };

  // Drop handlers
  const handleDragOver = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
  };

  const handleDrop = (e: DragEvent<HTMLDivElement>,isOriginal:boolean) => {
    e.preventDefault();
    e.stopPropagation();

    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      processFiles(e.dataTransfer.files,isOriginal);
      e.dataTransfer.clearData();
    }
  };


  // Input change handler
  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>,isOriginal:boolean) => {
    if (e.target.files && e.target.files.length > 0) {
      processFiles(e.target.files,isOriginal);
    }
  };

  const handleSubmit = async () => {
  if (original_images.length > 0 && styled_images.length > 0) {
    const formData = new FormData();
    original_images.forEach((img, i) => {
      if (img.file) {
        formData.append('original_images', img.file, img.id || `image_${i}.jpg`);
      }
    });
    styled_images.forEach((img,i)=>{
      if (img.file) {
        formData.append('styled_images',img.file,img.id || `image_${i}.jpg`);
      }
    })

    try {
      const res = await fetch('http://127.0.0.1:5000/train', {
        method: 'POST',
        body: formData,
      });

      const data = await res.json();
      console.log(data);
    } catch (error) {
      console.error('Upload error:', error);
    }
  }
};



  return (
    <div className="h-auto my-3 bg-gray-800 w-auto rounded-md border border-gray-500 mx-3 p-4">
      <div className="header text-white">
        <h1 className="text-xl flex gap-2 py-2 px-2">
          <img src={brain} alt="Brain Icon" className="w-6 h-6" />
          Make Your Custom Art Styler!
        </h1>
      </div>

      <div className="content text-white">
        <div className="input_header mb-4">
          <h2>Name</h2>
          <input
            type="text"
            placeholder="Enter the name of your style"
            className="mt-1 w-full bg-gray-700 border border-gray-600 rounded p-2"
          />
        </div>

        <div className="input_info mb-4">
          <p>Short name used to identify your style in Model</p>
          <p>Upload at least 10-15 images for concise results</p>
        </div>

        <div className="imageDropContainer flex gap-3">
          <div className="originalContainer">
          <div
          className="image_dropper border-dashed border-2 border-gray-500 rounded h-40 flex flex-col items-center justify-center"
          onDragOver={handleDragOver}
          onDrop={(e)=> handleDrop(e,true)}
        >
          <p>Drag & drop original images here, or</p>
          <label className="mt-2 inline-block cursor-pointer text-blue-400 hover:underline">
            Browse files
            <input
              type="file"
              multiple
              accept="image/*"
              onChange={(e)=> handleFileSelect(e,true)}
              className="hidden"
            />
          </label>
        </div>

        <div className="mt-4 grid grid-cols-3 gap-2 overflow-y-auto max-h-48">
          {original_images.map((img) => (
            <div key={img.id} className="relative">
              <img
                src={img.preview}
                alt={img.file.name}
                className="w-full h-20 object-cover rounded"
              />
            </div>
          ))}
        </div>
        </div>
        <div className="styleContainer">
          <div
          className="image_dropper border-dashed border-2 border-gray-500 rounded h-40 flex flex-col items-center justify-center"
          onDragOver={handleDragOver}
          onDrop={(e)=> handleDrop(e,false)}
        >
          <p>Drag & drop style images here, or</p>
          <label className="mt-2 inline-block cursor-pointer text-blue-400 hover:underline">
            Browse files
            <input
              type="file"
              multiple
              accept="image/*"
              onChange={(e)=> handleFileSelect(e,false)}
              className="hidden"
            />
          </label>
        </div>

        <div className="mt-4 grid grid-cols-3 gap-2 overflow-y-auto max-h-48">
          {styled_images.map((img) => (
            <div key={img.id} className="relative">
              <img
                src={img.preview}
                alt={img.file.name}
                className="w-full h-20 object-cover rounded"
              />
            </div>
          ))}
        </div>
        </div>
        </div>
        <div className="submit">
          <button className="bg-blue-500 text-white px-3 py-1 rounded-2xl mx-44 cursor-pointer" onClick={handleSubmit}>submit</button>
        </div>
      </div>
    </div>
  );
};

export default TrainerContainer;
