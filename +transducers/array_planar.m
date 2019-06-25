%
% superclass for all planar transducer arrays
%
% author: Martin F. Schiffner
% date: 2017-04-19
% modified: 2019-06-05
%
classdef array_planar < transducers.array

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% properties
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	properties (SetAccess = private)

        % dependent properties
        positions_ctr ( :, : ) physical_values.length	% discrete positions of the centroids

	end % properties

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% methods
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = array_planar( apertures )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure cell array for apertures
            if ~iscell( apertures )
                apertures = { apertures };
            end

            % ensure class transducers.face_planar
            indicator_nonplanar = cellfun( @( x ) ~isa( x, 'transducers.face_planar' ), apertures );
            if any( indicator_nonplanar( : ) )
                errorStruct.message = 'apertures must be transducers.face_planar!';
                errorStruct.identifier = 'array_planar:NoPlanarFaces';
                error( errorStruct );
            end

            %--------------------------------------------------------------
            % 2.) create planar transducer arrays
            %--------------------------------------------------------------
            % constructor of superclass
            objects@transducers.array( apertures );

            % iterate planar transducer arrays
            for index_object = 1:numel( objects )

                % set dependent properties
                objects( index_object ).positions_ctr = center( apertures{ index_object } );

            end % for index_object = 1:numel( objects )

        end % function objects = array_planar( apertures )

        %------------------------------------------------------------------
        % discretize
        %------------------------------------------------------------------
        function structs_out = discretize( arrays_planar, options_elements )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure class transducers.array_planar
            if ~isa( arrays_planar, 'transducers.array_planar' )
                errorStruct.message = 'arrays_planar must be transducers.array_planar!';
                errorStruct.identifier = 'discretize:NoPlanarArrays';
                error( errorStruct );
            end

            % method discretize in face ensures class discretizations.parameters for options_elements

            % multiple arrays_planar / scalar options_elements
            if ~isscalar( arrays_planar ) && isscalar( options_elements )
                options_elements = repmat( options_elements, size( arrays_planar ) );
            end

            % ensure equal number of dimensions and sizes
            auxiliary.mustBeEqualSize( arrays_planar, options_elements );

            %--------------------------------------------------------------
            % 2.) discretize planar transducer arrays
            %--------------------------------------------------------------
            % specify cell array for structs_out
            structs_out = cell( size( arrays_planar ) );

            % iterate planar transducer arrays
            for index_object = 1:numel( arrays_planar )

                % discretize aperture
                structs_out{ index_object } = discretize( arrays_planar( index_object ).aperture, options_elements( index_object ) );

            end % for index_object = 1:numel( arrays_planar )

            % avoid cell array for single planar transducer array
            if isscalar( arrays_planar )
                structs_out = structs_out{ 1 };
            end

        end % function structs_out = discretize( arrays_planar, options_elements )

	end % methods

end % classdef array_planar < transducers.array
