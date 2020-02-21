%
% abstract superclass for all spatial anti-aliasing filters
%
% author: Martin F. Schiffner
% date: 2019-07-11
% modified: 2020-02-21
%
classdef (Abstract) anti_aliasing < scattering.options.template

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% methods
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = anti_aliasing( size )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % superclass ensures row vector for size
            % superclass ensures nonempty positive integers for size

            %--------------------------------------------------------------
            % 2.) create spatial anti-aliasing filter options
            %--------------------------------------------------------------
            % constructor of superclass
            objects@scattering.options.template( size );

        end % function objects = anti_aliasing( size )

        %------------------------------------------------------------------
        % apply spatial anti-aliasing filters
        %------------------------------------------------------------------
        function hs_transfer = apply( filters, setups, hs_transfer, indices_element )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure class scattering.options.anti_aliasing
            if ~isa( filters, 'scattering.options.anti_aliasing' )
                errorStruct.message = 'filters must be scattering.options.anti_aliasing!';
                errorStruct.identifier = 'apply:NoSpatialAntiAliasingFilters';
                error( errorStruct );
            end

            % ensure class scattering.sequences.setups.setup
            if ~isa( setups, 'scattering.sequences.setups.setup' )
                errorStruct.message = 'setups must be scattering.sequences.setups.setup!';
                errorStruct.identifier = 'apply:NoSetups';
                error( errorStruct );
            end

            % ensure class scattering.sequences.setups.transducers.array_planar_regular_orthogonal
            indicator = cellfun( @( x ) ~isa( x, 'scattering.sequences.setups.transducers.array_planar_regular_orthogonal' ), { setups.xdc_array } );
            if any( indicator( : ) )
                errorStruct.message = 'setups.xdc_array must be scattering.sequences.setups.transducers.array_planar_regular_orthogonal!';
                errorStruct.identifier = 'apply:NoOrthogonalRegularPlanarArrays';
                error( errorStruct );
            end

            % ensure class processing.field
            if ~isa( hs_transfer, 'processing.field' )
                errorStruct.message = 'hs_transfer must be processing.field!';
                errorStruct.identifier = 'apply:NoFields';
                error( errorStruct );
            end

            % ensure nonempty indices_element
            if nargin < 4 || isempty( indices_element )
                indices_element = 1;
            end

            % ensure cell array for indices_element
            if ~iscell( indices_element )
                indices_element = { indices_element };
            end

            % ensure equal number of dimensions and sizes
            auxiliary.mustBeEqualSize( filters, setups, hs_transfer, indices_element );

            %--------------------------------------------------------------
            % 2.) apply spatial anti-aliasing filters
            %--------------------------------------------------------------
            % iterate spatial anti-aliasing filters
            for index_object = 1:numel( filters )

                %----------------------------------------------------------
                % a) apply spatial anti-aliasing filter (scalar)
                %----------------------------------------------------------
                hs_transfer( index_object ) = apply_scalar( filters( index_object ), setups( index_object ), hs_transfer( index_object ), indices_element{ index_object } );

            end % for index_object = 1:numel( filters )

        end % function hs_transfer = apply( filters, setups, hs_transfer, indices_element )

	end % methods

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% methods (Abstract)
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods (Abstract)

        %------------------------------------------------------------------
        % compute spatial anti-aliasing filters
        %------------------------------------------------------------------
% TODO: implement in superclass and reduce method in subclasses
        filters = compute_filter( options_anti_aliasing, flags )

	end % methods (Abstract)

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% methods (Abstract, protected, hidden)
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods (Abstract, Access = protected, Hidden)

        %------------------------------------------------------------------
        % apply spatial anti-aliasing filter (scalar)
        %------------------------------------------------------------------
        h_transfer = apply_scalar( filter, setup, h_transfer, index_element )

	end % methods (Abstract, Access = protected, Hidden)

end % classdef (Abstract) anti_aliasing < scattering.options.template
