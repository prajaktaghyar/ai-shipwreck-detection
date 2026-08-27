const form = document.getElementById("uploadForm");

const imageInput = document.getElementById("imageInput");

const loading = document.getElementById("loading");

const resultSection =
    document.getElementById("resultSection");

const resultImage =
    document.getElementById("resultImage");

const resultStatus =
    document.getElementById("resultStatus");

const resultClass =
    document.getElementById("resultClass");

const resultConfidence =
    document.getElementById("resultConfidence");

const detectionsContainer =
    document.getElementById("detections");


form.addEventListener(
    "submit",
    async function(event) {

        event.preventDefault();

        const file = imageInput.files[0];

        if (!file) {

            alert(
                "Please select a sonar image."
            );

            return;
        }


        const formData = new FormData();

        formData.append(
            "image",
            file
        );


        loading.classList.remove(
            "hidden"
        );

        resultSection.classList.add(
            "hidden"
        );


        try {

            const response = await fetch(
                "/api/detect",
                {
                    method: "POST",
                    body: formData
                }
            );


            const data =
                await response.json();


            if (!response.ok || !data.success) {

                throw new Error(
                    data.error ||
                    "Detection failed"
                );

            }


            // ------------------------------------------------
            // RESULT IMAGE
            // ------------------------------------------------

            if (data.result_image) {

                resultImage.src =
                    data.result_image +
                    "?t=" +
                    Date.now();

            }


            // ------------------------------------------------
            // BASIC INFORMATION
            // ------------------------------------------------

            resultStatus.textContent =
                data.status || "-";


            resultClass.textContent =
                data.best_class || "None";


            if (
                data.best_confidence !==
                undefined
            ) {

                resultConfidence.textContent =
                    data.best_confidence +
                    "%";

            } else {

                resultConfidence.textContent =
                    "0%";

            }


            // ------------------------------------------------
            // ALL DETECTIONS
            // ------------------------------------------------

            detectionsContainer.innerHTML =
                "";


            if (
                !data.detections ||
                data.detections.length === 0
            ) {

                detectionsContainer.innerHTML =
                    `
                    <div class="detection-item">
                        No objects detected.
                    </div>
                    `;

            } else {

                data.detections.forEach(
                    function(item, index) {

                        const box =
                            item.bbox;


                        const element =
                            document.createElement(
                                "div"
                            );


                        element.className =
                            "detection-item";


                        element.innerHTML = `

                            <strong>
                                Detection ${index + 1}
                            </strong>

                            <br><br>

                            Class:
                            ${item.class_name}

                            <br>

                            Confidence:
                            ${item.confidence_percent}%

                            <br>

                            Level:
                            ${item.confidence_level}

                            <br>

                            Status:
                            ${item.status}

                            <br>

                            Bounding Box:
                            (${box.x1},
                             ${box.y1},
                             ${box.x2},
                             ${box.y2})

                        `;


                        detectionsContainer.appendChild(
                            element
                        );

                    }
                );

            }


            resultSection.classList.remove(
                "hidden"
            );


        } catch (error) {

            alert(
                "Error: " +
                error.message
            );

        } finally {

            loading.classList.add(
                "hidden"
            );

        }

    }
);