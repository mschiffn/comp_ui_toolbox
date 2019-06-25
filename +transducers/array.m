%
% superclass for all transducer arrays
%
% author: Martin F. Schiffner
% date: 2017-04-20
% modified: 2019-06-04
%
classdef array

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% properties
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	properties (SetAccess = private)

        % independent properties
        aperture ( :, 1 ) transducers.face	% aperture is a column vector of vibrating faces

        % dependent properties
        N_dimensions ( 1, 1 ) double { mustBeInteger, mustBePositive, mustBeNonempty } = 2	% number of dimensions (1)
        N_elements ( 1, 1 ) double { mustBeInteger, mustBePositive, mustBeNonempty } = 128	% total number of elements (1)

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
            % 2.) create transducer arrays
            %--------------------------------------------------------------
            % repeat default transducer array
            objects = repmat( objects, size( apertures ) );

            % iterate transducer arrays
            for index_object = 1:numel( objects )

                % ensure class transducers.face
                if ~isa( apertures{ index_object }, 'transducers.face' )
                    errorStruct.message = sprintf( 'apertures{ %d } must be transducers.face!', index_object );
                    errorStruct.identifier = 'array:NoFaces';
                    error( errorStruct );
                end

                % ensure identical numbers of dimensions
                N_dimensions = get_N_dimensions( apertures{ index_object } );
                if ~all( N_dimensions( : ) == N_dimensions( 1 ) )
                    errorStruct.message = 'Numbers of dimensions of the vibrating faces do not match!';
                    errorStruct.identifier = 'array:DimensionMismatch';
                    error( errorStruct );
                end

                % set independent properties
                objects( index_object ).aperture = apertures{ index_object };

                % set dependent properties
                objects( index_object ).N_dimensions = objects( index_object ).aperture( 1 ).shape.N_dimensions;
                objects( index_object ).N_elements = numel( objects( index_object ).aperture );

            end % for index_object = 1:numel( objects )

        end % function objects = array( apertures )

	end % methods

end % classdef array
