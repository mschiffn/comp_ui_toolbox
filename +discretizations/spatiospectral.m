%
% superclass for all spatiospectral discretizations
%
% author: Martin F. Schiffner
% date: 2019-02-25
% modified: 2019-07-15
%
classdef spatiospectral

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% properties
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	properties (SetAccess = private)

        % independent properties
        spatial ( 1, 1 ) discretizations.spatial        % spatial discretization
        spectral ( :, 1 ) discretizations.spectral      % spectral discretization

        

    end % properties

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% methods
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = spatiospectral( spatials, spectrals )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % return empty object if no arguments
            if nargin == 0
                return;
            end

            % ensure cell array for spectrals
            if ~iscell( spectrals )
                spectrals = { spectrals };
            end

            % ensure equal number of dimensions and sizes
            auxiliary.mustBeEqualSize( spatials, spectrals );

            %--------------------------------------------------------------
            % 2.) create spatiospectral discretizations
            %--------------------------------------------------------------
            % repeat default spatiospectral discretization
            objects = repmat( objects, size( spatials ) );

            % iterate spatiospectral discretizations
            for index_object = 1:numel( spatials )

% TODO: check for valid spatial discretization (sampling theorem)

                %----------------------------------------------------------
                % a) set independent properties
                %----------------------------------------------------------
                objects( index_object ).spatial = spatials( index_object );
                objects( index_object ).spectral = spectrals{ index_object };

                %----------------------------------------------------------
                % b) set dependent properties
                %----------------------------------------------------------
                
                

            end % for index_object = 1:numel( spatials )

        end % function objects = spatiospectral( spatials, spectrals )

        

	end % methods

end % classdef spatiospectral
