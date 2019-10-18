%
% superclass for all planar transducer arrays
%
% author: Martin F. Schiffner
% date: 2019-08-16
% modified: 2019-10-17
%
classdef array_planar < scattering.sequences.setups.transducers.array

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

            % ensure class scattering.sequences.setups.transducers.face_planar
            indicator_nonplanar = cellfun( @( x ) ~isa( x, 'scattering.sequences.setups.transducers.face_planar' ), apertures );
            if any( indicator_nonplanar( : ) )
                errorStruct.message = 'apertures must be scattering.sequences.setups.transducers.face_planar!';
                errorStruct.identifier = 'array_planar:NoPlanarFaces';
                error( errorStruct );
            end

            %--------------------------------------------------------------
            % 2.) create continuous planar transducer arrays
            %--------------------------------------------------------------
            % constructor of superclass
            objects@scattering.sequences.setups.transducers.array( apertures );

            % iterate planar transducer arrays
            for index_object = 1:numel( objects )

                % set dependent properties
                objects( index_object ).positions_ctr = center( apertures{ index_object } );

            end % for index_object = 1:numel( objects )

        end % function objects = array_planar( apertures )

	end % methods

end % classdef array_planar < scattering.sequences.setups.transducers.array
