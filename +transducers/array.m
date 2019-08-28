%
% superclass for all transducer arrays
%
% author: Martin F. Schiffner
% date: 2019-08-16
% modified: 2019-08-27
%
classdef array

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% properties
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	properties (SetAccess = private)

        % independent properties
        aperture ( :, 1 ) transducers.face { mustBeNonempty } = transducers.face	% aperture is a column vector of vibrating faces

        % dependent properties
        N_dimensions ( 1, 1 ) double { mustBeInteger, mustBePositive, mustBeNonempty } = 2	% number of dimensions (1)
        N_elements ( 1, 1 ) double { mustBeInteger, mustBePositive, mustBeNonempty } = 1	% total number of elements (1)

    end % properties

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% methods
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = array( apertures )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure cell array for apertures
            if ~iscell( apertures )
                apertures = { apertures };
            end

            %--------------------------------------------------------------
            % 2.) create continuous transducer arrays
            %--------------------------------------------------------------
            % repeat default transducer array
            objects = repmat( objects, size( apertures ) );

            % iterate continuous transducer arrays
            for index_object = 1:numel( objects )

                % ensure class transducers.face (column vector)
                if ~( isa( apertures{ index_object }, 'transducers.face' ) && iscolumn( apertures{ index_object } ) )
                    errorStruct.message = sprintf( 'apertures{ %d } must be a column vector of transducers.face!', index_object );
                    errorStruct.identifier = 'array:NoFaces';
                    error( errorStruct );
                end

                % ensure identical numbers of dimensions
                N_dimensions_act = get_N_dimensions( apertures{ index_object } );
                if ~all( N_dimensions_act( : ) == N_dimensions_act( 1 ) )
                    errorStruct.message = 'Numbers of dimensions of the vibrating faces do not match!';
                    errorStruct.identifier = 'array:DimensionMismatch';
                    error( errorStruct );
                end
% check for continuous shapes
% TODO: prevent overlapping of faces

                % set independent properties
                objects( index_object ).aperture = apertures{ index_object };

                % set dependent properties
                objects( index_object ).N_dimensions = N_dimensions_act( 1 );
                objects( index_object ).N_elements = numel( objects( index_object ).aperture );

            end % for index_object = 1:numel( objects )

        end % function objects = array( apertures )

        %------------------------------------------------------------------
        % discretize
        %------------------------------------------------------------------
        function arrays = discretize( arrays, methods_faces )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure class transducers.array
            if ~isa( arrays, 'transducers.array' )
                errorStruct.message = 'arrays must be transducers.array!';
                errorStruct.identifier = 'discretize:NoArrays';
                error( errorStruct );
            end

            % method discretize in transducers.face ensures shapes and class discretizations.options_spatial_method for methods_faces

            % multiple arrays / single methods_faces
            if ~isscalar( arrays ) && isscalar( methods_faces )
                methods_faces = repmat( methods_faces, size( arrays ) );
            end

            % single arrays / multiple methods_faces
            if isscalar( arrays ) && ~isscalar( methods_faces )
                arrays = repmat( arrays, size( methods_faces ) );
            end

            % ensure equal number of dimensions and sizes
            auxiliary.mustBeEqualSize( arrays, methods_faces );

            %--------------------------------------------------------------
            % 2.) discretize continuous transducer arrays
            %--------------------------------------------------------------
            % iterate continuous transducer arrays
            for index_object = 1:numel( arrays )

                % discretize aperture
                arrays( index_object ).aperture = discretize( arrays( index_object ).aperture, methods_faces( index_object ) );

            end % for index_object = 1:numel( arrays )

        end % function arrays = discretize( arrays, methods_faces )

	end % methods

end % classdef array
