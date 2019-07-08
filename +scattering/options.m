%
% superclass for all scattering operator options
%
% author: Martin F. Schiffner
% date: 2019-02-15
% modified: 2019-04-13
%
classdef options

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% public properties
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	properties (SetAccess = public)

        % independent properties
        spatial_aliasing ( 1, 1 ) scattering.options_aliasing = scattering.options_aliasing.include     % aliasing
        gpu_index ( 1, 1 ) double { mustBeInteger, mustBeNonnegative, mustBeNonempty } = 0              % index of GPU

    end % properties

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% private properties
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	properties (SetAccess = private)

        % independent properties
        discretization ( 1, 1 ) discretizations.options = discretizations.options                       % spatiospectral discretization
        materials ( 1, 1 ) scattering.options_material = scattering.options_material.compressibility	% material parameters

    end % properties

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% methods
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function object = options( varargin )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % check number of arguments
            if nargin == 0
                return;
            end

            %--------------------------------------------------------------
            % 2.) parse arguments
            %--------------------------------------------------------------
            for index_arg = 1:nargin

                switch class( varargin{ index_arg } )

                    %------------------------------------------------------
                    % spatiospectral discretization
                    %------------------------------------------------------
                    case 'discretizations.options'
                        object.discretization = varargin{ index_arg };

                    %------------------------------------------------------
                    % anti-aliasing options
                    %------------------------------------------------------
                    case 'scattering.options_aliasing'
                        object.spatial_aliasing = varargin{ index_arg };

                    %------------------------------------------------------
                    % material parameters
                    %------------------------------------------------------
                    case 'scattering.options_material'
                        object.materials = varargin{ index_arg };

                    %------------------------------------------------------
                    % unknown
                    %------------------------------------------------------
                    otherwise
                        errorStruct.message = sprintf( 'Class of varargin{ %d } is unknown!', index_arg );
                        errorStruct.identifier = 'options:UnknownClass';
                        error( errorStruct );

                end % switch class( varargin{ index_arg } )

            end % for index_arg = 1:nargin

        end % function object = options( varargin )
% 
%         %------------------------------------------------------------------
%         % toggle spatial_aliasing
%         %------------------------------------------------------------------
%         function set_spatial_aliasing( options )
% 
%             %--------------------------------------------------------------
%             % 1.) check arguments
%             %--------------------------------------------------------------
%             % ensure class scattering.options
%             if ~isa()
%             end
% 
%             %--------------------------------------------------------------
%             % 2.)
%             %--------------------------------------------------------------
%             for index_object = 1:numel( options )
%             end
% 
%         end % function set_spatial_aliasing( options )

	end % methods

end % classdef options
